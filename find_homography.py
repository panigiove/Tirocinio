import json, numpy as np, cv2

corr_path = r"court_points_corrispondency.json"
cam4_path = r"cam4_img_corners_rectified.json"
cam13_path = r"cam13_img_corners_rectified.json"

output_path = r"homography_cam4_to_cam13.json"

corr = json.load(open(corr_path))["rectified_img_corners_corrispondency"]
cam4 = {p["index"]: p["point"] for p in json.load(open(cam4_path))["rectified_img_corners_indexed"]}
cam13 = {p["index"]: p["point"] for p in json.load(open(cam13_path))["rectified_img_corners_indexed"]}

src = []
dst = []
for pair in corr:
    p4 = cam4[pair["point_id_cam4"]]
    p13 = cam13[pair["point_id_cam13"]]
    if any(np.isnan(p4)) or any(np.isnan(p13)):
        continue
    src.append(p4)
    dst.append(p13)

src = np.array(src, dtype=np.float32)
dst = np.array(dst, dtype=np.float32)

H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC)
H_inv = np.linalg.inv(H)

print("num points", len(src))
print("H cam4->cam13")
print(H)
print("\nH cam13->cam4 (inverse)")
print(H_inv)
print("\nRANSAC inliers", int(mask.sum()) if mask is not None else None)

payload = {
    "src": "cam4",
    "dst": "cam13",
    "method": "RANSAC",
    "num_points": int(len(src)),
    "inliers": int(mask.sum()) if mask is not None else None,
    "H": H.tolist(),
    "H_inv": H_inv.tolist(),
}
with open(output_path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"\nSaved homography to {output_path}")
