from deepbrain import Extractor
import numpy as np, os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

ext = Extractor()
base_path = "/home/workspace/K-Alzheimer/ALL_DATASETS/dataset_adni_good"
seg_base_path = "/home/workspace/K-Alzheimer/ALL_DATASETS/dataset_adni_segmented"

def process_file(paths):
    in_path, out_path = paths
    vol = np.load(in_path)["volume"]
    prob = ext.run(vol)
    mask = prob > 0.5
    seg = (vol * mask).astype(np.float32)
    scale = 255.0 / (seg.max() + 1e-8)
    seg_scaled = (seg * scale).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, volume=seg_scaled)


if __name__ == "__main__":
    file_list = []
    for disease in ["AD", "CN", "MCI"]:
        src_dir = os.path.join(base_path, disease)
        dst_dir = os.path.join(seg_base_path, disease)
        for f in os.listdir(src_dir):
            if f.endswith(".npz"):
                file_list.append((os.path.join(src_dir, f), os.path.join(dst_dir, f)))

    with Pool(processes=cpu_count() // 2) as pool:
        for _ in tqdm(pool.imap_unordered(process_file, file_list),
                      total=len(file_list), desc="Segmenting"):
            pass
