from pyspark.sql import DataFrame
from pyspark.sql.functions import col

def compute_real_world_area(inference_df: DataFrame, spatial_resolution_meters: float = 0.5) -> DataFrame:
    """
    Computes change metrics in physical units (square meters and hectares).
    Default spatial_resolution_meters = 0.5m/pixel (LEVIR-CD standard resolution).
    """
    pixel_area_m2 = spatial_resolution_meters * spatial_resolution_meters

    # Add calculated area columns using Spark expressions
    metrics_df = inference_df \
        .withColumn("changed_area_sqm", col("changed_pixels") * pixel_area_m2) \
        .withColumn("changed_area_hectares", col("changed_area_sqm") / 10000.0)

    return metrics_df