import os
import cv2
import numpy as np
import torch
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from src.models.network import SiameseUNet

def load_bitemporal_image_pairs(spark: SparkSession, dataset_root: str) -> DataFrame:
    dir_a = os.path.join(dataset_root, "A")
    dir_b = os.path.join(dataset_root, "B")
    
    if not os.path.exists(dir_a) or not os.path.exists(dir_b):
        raise FileNotFoundError(f"Missing dataset folders at {dir_a} or {dir_b}")
        
    filenames_a = sorted(os.listdir(dir_a))
    image_pairs = []
    
    for fname in filenames_a:
        if fname.lower().endswith(('.png', '.tif', '.tiff', '.jpg')):
            path_a = os.path.join(dir_a, fname)
            path_b = os.path.join(dir_b, fname)
            
            if os.path.exists(path_b):
                tile_id = os.path.splitext(fname)[0]
                image_pairs.append((tile_id, path_a, path_b))

    schema = StructType([
        StructField("tile_id", StringType(), False),
        StructField("image_t1_path", StringType(), False),
        StructField("image_t2_path", StringType(), False)
    ])

    return spark.createDataFrame(image_pairs, schema=schema)


def _process_partition(iterator, model_weights_path: str):
    device = torch.device("cpu")
    
    # Load model once per worker partition
    model = SiameseUNet(in_channels=3, out_channels=1).to(device)
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model.eval()

    for row in iterator:
        tile_id = row.tile_id
        path_a = row.image_t1_path
        path_b = row.image_t2_path

        img1 = cv2.imread(path_a)
        img2 = cv2.imread(path_b)

        if img1 is None or img2 is None:
            continue

        img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        t1_tensor = torch.from_numpy(img1_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t2_tensor = torch.from_numpy(img2_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        t1_tensor = t1_tensor.to(device)
        t2_tensor = t2_tensor.to(device)

        with torch.no_grad():
            logits = model(t1_tensor, t2_tensor)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).squeeze().cpu().numpy().astype(np.uint8)

        total_pixels = preds.size
        changed_pixels = int(np.sum(preds == 1))
        change_ratio = float(changed_pixels / total_pixels) if total_pixels > 0 else 0.0
        mean_confidence = float(probs.squeeze().cpu().numpy().mean())

        yield (tile_id, path_a, path_b, total_pixels, changed_pixels, change_ratio, mean_confidence)


def run_distributed_inference(spark: SparkSession, pairs_df: DataFrame, model_path: str = "models/siamese_unet.pth") -> DataFrame:
    abs_model_path = os.path.abspath(model_path)

    output_schema = StructType([
        StructField("tile_id", StringType(), False),
        StructField("image_t1_path", StringType(), False),
        StructField("image_t2_path", StringType(), False),
        StructField("total_pixels", IntegerType(), False),
        StructField("changed_pixels", IntegerType(), False),
        StructField("change_ratio", DoubleType(), False),
        StructField("mean_confidence", DoubleType(), False)
    ])

    rdd_results = pairs_df.rdd.mapPartitions(lambda iter_: _process_partition(iter_, abs_model_path))
    return spark.createDataFrame(rdd_results, schema=output_schema)
