import pandas as pd
import geopandas as gpd
import numpy as np
import h3
from scipy.spatial import cKDTree

def calculate_macro_scores(hex_gdf: gpd.GeoDataFrame, 
                           pvz_gdf: gpd.GeoDataFrame, 
                           realty_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Рассчитывает индекс дефицита для каждого гексагона на основе 
    удаленности от ПВЗ и наличия коммерческой недвижимости.
    """
    scored_hex = hex_gdf.copy()
    
    original_crs = scored_hex.crs
    
    scored_hex_m = scored_hex.to_crs(epsg=3857)
    pvz_gdf_m = pvz_gdf.to_crs(epsg=3857)
    realty_gdf_m = realty_gdf.to_crs(epsg=3857)
    
    pvz_coords = np.array(list(zip(pvz_gdf_m.geometry.x, pvz_gdf_m.geometry.y)))
    hex_centroids = scored_hex_m.geometry.centroid
    hex_coords = np.array(list(zip(hex_centroids.x, hex_centroids.y)))
    
    tree = cKDTree(pvz_coords)
    distances, _ = tree.query(hex_coords)
    
    decay_function = np.exp(-distances / 500.0)
    
    population = scored_hex_m.get('population', 1) 
    
    unmet_demand_index = population * (1 - decay_function)
    scored_hex['unmet_demand'] = unmet_demand_index
    
    realty_in_hex = gpd.sjoin(realty_gdf_m, scored_hex_m[['hex_id', 'geometry']], how='inner', predicate='intersects')
    
    valid_hex_ids = realty_in_hex['hex_id'].unique()
    
    scored_hex['has_realty'] = scored_hex['hex_id'].isin(valid_hex_ids).astype(int)
    
    scored_hex['deficit_score'] = scored_hex['unmet_demand'] * scored_hex['has_realty']
    
    return scored_hex.to_crs(original_crs)


def select_top_5_hexagons_nms(scored_hex: gpd.GeoDataFrame, 
                              cannibalization_k_ring: int = 1) -> gpd.GeoDataFrame:
    """
    Выбирает ровно 5 лучших гексагонов, избегая их концентрации в одном месте 
    с помощью жадного алгоритма и пространственного штрафа (NMS).
    """
    candidates = scored_hex[scored_hex['deficit_score'] > 0].copy()
    
    selected_hex_ids = []
    
    for _ in range(5):
        if candidates.empty:
            break
            
        best_hex = candidates.loc[candidates['deficit_score'].idxmax()]
        best_hex_id = best_hex['hex_id']
        selected_hex_ids.append(best_hex_id)
        
        try:
            neighbors = h3.k_ring(best_hex_id, cannibalization_k_ring)
        except AttributeError:
            neighbors = h3.grid_disk(best_hex_id, cannibalization_k_ring)
            
        candidates = candidates[~candidates['hex_id'].isin(neighbors)]
        
    top_5_gdf = scored_hex[scored_hex['hex_id'].isin(selected_hex_ids)].copy()
    
    top_5_gdf = top_5_gdf.sort_values(by='deficit_score', ascending=False).reset_index(drop=True)
    
    return top_5_gdf

def evaluate_micro_candidates(top_hex_gdf: gpd.GeoDataFrame, 
                              realty_gdf: gpd.GeoDataFrame, 
                              pvz_gdf: gpd.GeoDataFrame, 
                              city_hex_gdf: gpd.GeoDataFrame, 
                              radius_m: int = 500) -> gpd.GeoDataFrame:
    realty_m = realty_gdf.to_crs(epsg=3857)
    pvz_m = pvz_gdf.to_crs(epsg=3857)
    hex_m = city_hex_gdf.to_crs(epsg=3857)
    top_hex_m = top_hex_gdf.to_crs(epsg=3857)

    candidates_m = gpd.sjoin(realty_m, top_hex_m[['hex_id', 'geometry']], how='inner', predicate='intersects')
    if 'index_right' in candidates_m.columns:
        candidates_m = candidates_m.drop(columns=['index_right'])
    candidates_m = candidates_m.drop_duplicates(subset=['id']).copy()

    pvz_buffers = pvz_m.geometry.buffer(radius_m)
    existing_coverage = pvz_buffers.unary_union

    def calc_mci_areal(geom):
        cand_buffer = geom.buffer(radius_m)
        new_area = cand_buffer.difference(existing_coverage)
        
        intersecting_hex = hex_m[hex_m.geometry.intersects(new_area)].copy()
        
        if intersecting_hex.empty:
            return 0.0
            
        intersect_areas = intersecting_hex.geometry.intersection(new_area).area
        coverage_ratios = intersect_areas / intersecting_hex.geometry.area
        marginal_pop = (intersecting_hex['population'] * coverage_ratios).sum()
        
        return marginal_pop

    candidates_m['marginal_coverage'] = candidates_m.geometry.apply(calc_mci_areal)
    
    candidates_m['cost_efficiency'] = candidates_m['price_rub_per_month'] / candidates_m['marginal_coverage'].replace(0, 0.0001)

    return candidates_m.to_crs(realty_gdf.crs)

def business_roi_strategy(candidates_gdf: gpd.GeoDataFrame, 
                          ltv_estimate: float = 1500.0) -> gpd.GeoDataFrame:
    df = candidates_gdf.copy()
    
    df['expected_revenue'] = df['marginal_coverage'] * ltv_estimate
    df['business_roi'] = df['expected_revenue'] / df['price_rub_per_month']
    
    top_candidates = df.sort_values(['hex_id', 'business_roi'], ascending=[True, False])
    best_locations = top_candidates.groupby('hex_id').head(1).copy()
    
    return best_locations.sort_values('business_roi', ascending=False).reset_index(drop=True)

def calculate_business_impact(city_hex_gdf: gpd.GeoDataFrame,
                              pvz_gdf: gpd.GeoDataFrame,
                              final_locations_gdf: gpd.GeoDataFrame,
                              radius_m: int = 500) -> dict:
    hex_m = city_hex_gdf.to_crs(epsg=3857)
    pvz_m = pvz_gdf.to_crs(epsg=3857)
    final_m = final_locations_gdf.to_crs(epsg=3857)

    initial_buffers = pvz_m.geometry.buffer(radius_m)
    initial_coverage_area = initial_buffers.unary_union

    final_buffers = pd.concat([pvz_m.geometry, final_m.geometry]).buffer(radius_m)
    final_coverage_area = final_buffers.unary_union

    total_population = hex_m['population'].sum()

    def get_covered_pop(coverage_area):
        intersecting_hex = hex_m[hex_m.geometry.intersects(coverage_area)].copy()
        if intersecting_hex.empty:
            return 0
        intersect_areas = intersecting_hex.geometry.intersection(coverage_area).area
        ratios = intersect_areas / intersecting_hex.geometry.area
        return (intersecting_hex['population'] * ratios).sum()

    initial_covered_pop = get_covered_pop(initial_coverage_area)
    final_covered_pop = get_covered_pop(final_coverage_area)

    audience_lift = final_covered_pop - initial_covered_pop
    total_rent = final_m['price_rub_per_month'].sum()
    cac = total_rent / audience_lift if audience_lift > 0 else 0

    initial_uncovered_pct = (total_population - initial_covered_pop) / total_population * 100
    final_uncovered_pct = (total_population - final_covered_pop) / total_population * 100
    blind_spot_reduction = initial_uncovered_pct - final_uncovered_pct

    return {
        'total_audience_lift': round(audience_lift),
        'total_rent_rub': round(total_rent),
        'cac_rub_per_person': round(cac, 2),
        'initial_uncovered_pct': round(initial_uncovered_pct, 2),
        'final_uncovered_pct': round(final_uncovered_pct, 2),
        'blind_spot_reduction_pct': round(blind_spot_reduction, 2)
    }
