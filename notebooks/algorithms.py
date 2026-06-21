import pandas as pd
import geopandas as gpd

REQUIRED_COLUMNS = [
    'h3_id', 
    'address', 
    'geometry', 
    'area_m2', 
    'price_rub_per_month', 
    'expected_coverage'
]

def get_best_locations(algorithm_func, realty_gdf, pvz_gdf, top_n=5, **kwargs):
    print(f"Запуск алгоритма: {algorithm_func.__name__}...")
    
    result_gdf = algorithm_func(realty_gdf, pvz_gdf, top_n, **kwargs)
    
    if not isinstance(result_gdf, gpd.GeoDataFrame):
        raise TypeError("Алгоритм должен возвращать GeoDataFrame!")
        
    missing_cols = set(REQUIRED_COLUMNS) - set(result_gdf.columns)
    if missing_cols:
        raise ValueError(f"Алгоритм не вернул обязательные колонки: {missing_cols}")
        
    print(f"Успех. Найдено точек: {len(result_gdf)}\n")
    return result_gdf.copy()


def naive_strategy(realty_gdf, pvz_gdf, top_n=5):
    """
    АЛГОРИТМ 1: Наивный
    """
    candidates = realty_gdf[realty_gdf['population_estimated'] > 0].copy()
    
    candidates = candidates.sort_values(by=['population_estimated', 'price_rub_per_month'], ascending=[False, True])
    
    candidates = candidates.drop_duplicates(subset=['address'])
    
    best_spots = candidates.head(top_n).copy()
    best_spots['expected_coverage'] = best_spots['population_estimated']
    
    return best_spots


def greedy_coverage_strategy(realty_gdf, pvz_gdf, top_n=5):
    """
    АЛГОРИТМ 2: Умный (Жадный)
    Учитывает каннибализацию. Каждая новая точка вычитает "своих" людей
    из общей базы, заставляя следующие точки открываться в других районах
    """
    COVERAGE_RADIUS = 500
    
    if not pvz_gdf.empty:
        existing_coverage = pvz_gdf.geometry.buffer(COVERAGE_RADIUS).unary_union
    else:
        from shapely.geometry import Polygon
        existing_coverage = Polygon()

    top_candidates = []
    working_realty = realty_gdf.copy()
    
    for step in range(top_n):
        best_score = -1
        best_idx = None
        best_net_pop = 0
        
        for idx, candidate in working_realty.iterrows():
            cand_buffer = candidate.geometry.buffer(COVERAGE_RADIUS)
            unique_area = cand_buffer.difference(existing_coverage)
            ratio = unique_area.area / cand_buffer.area if cand_buffer.area > 0 else 0
            net_new_pop = candidate['population_estimated'] * ratio
            score = net_new_pop 
            
            if score > best_score:
                best_score = score
                best_idx = idx
                best_net_pop = net_new_pop
                
        if best_idx is None or best_score <= 0:
            print("Больше нет точек с положительным приростом.")
            break

        winner = working_realty.loc[best_idx].copy()
        winner['expected_coverage'] = best_net_pop
        top_candidates.append(winner)

        existing_coverage = existing_coverage.union(winner.geometry.buffer(COVERAGE_RADIUS))
        working_realty = working_realty.drop(best_idx)
        
    return gpd.GeoDataFrame(top_candidates, crs=realty_gdf.crs)

def business_roi_strategy(realty_gdf, pvz_gdf, top_n=5):
    """
    АЛГОРИТМ 3: Многокритериальный (Бизнес-ROI)
    Ищет идеальный баланс между охватом новой аудитории и стоимостью аренды.
    Целевая метрика: Максимизация "Человек на 1000 вложенных рублей" с учетом пешей доступности.
    """
    COVERAGE_RADIUS = 500
    
    if not pvz_gdf.empty:
        existing_coverage = pvz_gdf.geometry.buffer(COVERAGE_RADIUS).unary_union
    else:
        from shapely.geometry import Polygon
        existing_coverage = Polygon()

    top_candidates = []
    working_realty = realty_gdf.copy()
    
    for step in range(top_n):
        best_score = -1
        best_idx = None
        best_net_pop = 0
        
        for idx, candidate in working_realty.iterrows():
            cand_buffer = candidate.geometry.buffer(COVERAGE_RADIUS)
            unique_area = cand_buffer.difference(existing_coverage)
            ratio = unique_area.area / cand_buffer.area if cand_buffer.area > 0 else 0
            
            # 1. Считаем чистый приток людей
            net_new_pop = candidate['population_estimated'] * ratio
            
            # 2. Получаем цену и площадь (защита от деления на ноль)
            price = max(candidate['price_rub_per_month'], 1)
            area = candidate['area_m2']
            
            # 3. ВЫЧИСЛЯЕМ БИЗНЕС-СКОР (МАГИЯ ЗДЕСЬ)
            # База: Сколько людей мы получаем на 1000 рублей аренды
            roi_score = (net_new_pop / price) * 1000
            
            # Бонус за площадь: ПВЗ меньше 40 квадратов - тесновато, даем штраф. 
            # Идеально 70-90 квадратов.
            if area < 40:
                roi_score *= 0.8  # Режем скор на 20%
            elif 60 <= area <= 100:
                roi_score *= 1.2  # Даем бонус 20% за идеальный размер
                
            # Итоговый балл
            score = roi_score 
            
            if score > best_score and net_new_pop > 50: # Отсекаем точки с нулевым или смешным приростом
                best_score = score
                best_idx = idx
                best_net_pop = net_new_pop
                
        if best_idx is None:
            print("Больше нет рентабельных точек для открытия.")
            break

        winner = working_realty.loc[best_idx].copy()
        winner['expected_coverage'] = best_net_pop
        
        # Можем сохранить сам скор для аналитики
        winner['business_score'] = best_score 
        
        top_candidates.append(winner)

        existing_coverage = existing_coverage.union(winner.geometry.buffer(COVERAGE_RADIUS))
        working_realty = working_realty.drop(best_idx)
        
    return gpd.GeoDataFrame(top_candidates, crs=realty_gdf.crs)
