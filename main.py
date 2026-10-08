import os
import sys
from src.spark.session import init_geoai_spark_session
from src.spark.inference import load_bitemporal_image_pairs, run_distributed_inference
from src.utils.spatial_metrics import compute_real_world_area

def main():
    print("==========================================================================")
    print("=== [Step 1/5] Initializing Apache Sedona & PySpark Distributed Engine ===")
    print("==========================================================================")
    spark = init_geoai_spark_session(driver_memory="8g")

    # 1. Verify Model Checkpoint Exists
    model_weights_path = os.path.abspath("models/siamese_unet.pth")
    if not os.path.exists(model_weights_path):
        print(f"[ERROR] Model weights file not found at {model_weights_path}")
        print("Run step 2 checkpoint generation first or verify the path.")
        spark.stop()
        sys.exit(1)

    # 2. Ingest Dataset
    dataset_path = os.path.join("data", "raw", "LEVIR-CD", "test")

    if os.path.exists(dataset_path):
        print("\n==========================================================================")
        print("=== [Step 1.2] Scanning & Ingesting Bitemporal Image Pairs ===")
        print("==========================================================================")
        pairs_df = load_bitemporal_image_pairs(spark, dataset_path)
        
        print("\n[PREVIEW] Ingested Satellite Image Pairs:")
        pairs_df.show(5, truncate=False)

        print("\n==========================================================================")
        print("=== [Step 3/5] Executing Parallelized PyTorch Inference via mapPartitions ===")
        print("==========================================================================")
        inference_df = run_distributed_inference(spark, pairs_df, model_weights_path)

        print("\n==========================================================================")
        print("=== [Step 4/5] Computing Geospatial Surface Area Metrics (m² & Hectares) ===")
        print("==========================================================================")
        # LEVIR-CD standard resolution is 0.5 meters per pixel
        metrics_df = compute_real_world_area(inference_df, spatial_resolution_meters=0.5)

        print("\n[PREVIEW] Distributed Inference Results & Surface Metrics:")
        metrics_df.select(
            "tile_id", 
            "total_pixels", 
            "changed_pixels", 
            "change_ratio", 
            "changed_area_sqm", 
            "changed_area_hectares", 
            "mean_confidence"
        ).show(10, truncate=False)

        print("\n==========================================================================")
        print("=== [Step 5/5] Exporting Analytics Data for Power BI ===")
        print("==========================================================================")
        os.makedirs(os.path.join("data", "processed"), exist_ok=True)
        output_csv_path = os.path.join("data", "processed", "metrics_summary.csv")

        # Export consolidated metrics table to CSV for Power BI visualization
        metrics_pandas_df = metrics_df.toPandas()
        metrics_pandas_df.to_csv(output_csv_path, index=False)
        print(f"[SUCCESS] Exported summary metrics ({len(metrics_pandas_df)} tiles) to: {output_csv_path}")

    else:
        print(f"\n[NOTICE] Dataset directory not found at '{dataset_path}'.")
        print("Please ensure images are placed inside data/raw/LEVIR-CD/train/A/ and B/")

    # Clean shutdown
    spark.stop()
    print("\n[SUCCESS] GeoAI Change Detection Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
