"""
One-off patch for pupil-labs-dynamic-rim (yurahwang97 fork, commit a017966).

Bug: save_videos() draws labels with cv2.putText() directly onto `bkg`, which
is created via np.zeros(...) => float64. cv2.putText requires a CV_8U image,
so this crashes with:
    cv2.error: ... (-215:Assertion failed) img.depth() == CV_8U in function 'putText'

`bkg` is only ever converted to uint8 *after* the labels block (via
cv2.normalize(..., cv2.CV_8U)), one step too late.

Fix: do the normalize-to-uint8 step first, draw labels on the resulting
uint8 image, and drop the now-redundant duplicate normalize call. This
preserves the exact same NORM_MINMAX contrast stretch, just computed a
few lines earlier, and keeps `bkg` itself float64 for blending in the
next loop iteration.

Verified: reproduced the exact crash in a sandbox with the same
opencv-python==5.0.0.93, confirmed this patch resolves it and the
pipeline proceeds well past that point.

Usage:
    python3 patch_labels_dtype.py
"""
import sys

path = "/Users/xuebeichen/miniconda3/envs/pupil-cloud-local/lib/python3.11/site-packages/pupil_labs/dynamic_content_on_rim/dynamic_rim.py"

old = '''            if args.labels:
                # Add text to the frames
                y, w, h = 60, 360, 60
                label_device = ""
                with open(os.path.join(args.raw_folder_path, "info.json"), "r") as f:
                    import json

                    info = json.load(f)
                    # check if there is a frame_name field in the info.json file
                    label_device = "Neon" if "frame_name" in info else "Pupil Invisible"

                labels = [
                    {"text": label_device, "margin": 10},
                    {"text": "Reference Image", "margin": 10 + _etframe.width},
                    {
                        "text": "Screen Video",
                        "margin": 10 + (_etframe.width + refimg_finalwidth),
                    },
                ]
                for label in labels:
                    overlay = bkg[
                        y : y + h, (label["margin"] - 10) : (label["margin"] - 10 + w)
                    ]
                    whiterect = np.ones(overlay.shape) * 255
                    res = cv2.addWeighted(overlay, 0.5, whiterect, 0.5, 0)
                    bkg[
                        y : y + h, (label["margin"] - 10) : (label["margin"] - 10) + w
                    ] = res
                    bkg = cv2.putText(
                        bkg,
                        label["text"],
                        (label["margin"], 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 0),
                        2,
                        2,
                    )
            # Get the final frame
            out_ = bkg.copy()
            out_ = cv2.normalize(out_, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
'''

new = '''            # Get the final frame (normalize to 8-bit *before* drawing labels:
            # cv2.putText requires CV_8U, but bkg is float64)
            out_ = cv2.normalize(bkg, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            if args.labels:
                # Add text to the frames
                y, w, h = 60, 360, 60
                label_device = ""
                with open(os.path.join(args.raw_folder_path, "info.json"), "r") as f:
                    import json

                    info = json.load(f)
                    # check if there is a frame_name field in the info.json file
                    label_device = "Neon" if "frame_name" in info else "Pupil Invisible"

                labels = [
                    {"text": label_device, "margin": 10},
                    {"text": "Reference Image", "margin": 10 + _etframe.width},
                    {
                        "text": "Screen Video",
                        "margin": 10 + (_etframe.width + refimg_finalwidth),
                    },
                ]
                for label in labels:
                    overlay = out_[
                        y : y + h, (label["margin"] - 10) : (label["margin"] - 10 + w)
                    ]
                    whiterect = (np.ones(overlay.shape) * 255).astype(np.uint8)
                    res = cv2.addWeighted(overlay, 0.5, whiterect, 0.5, 0)
                    out_[
                        y : y + h, (label["margin"] - 10) : (label["margin"] - 10) + w
                    ] = res
                    out_ = cv2.putText(
                        out_,
                        label["text"],
                        (label["margin"], 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 0),
                        2,
                        2,
                    )
'''

src = open(path).read()
count = src.count(old)
print("occurrences found:", count)
if count != 1:
    print("ABORT: expected exactly 1 occurrence - file may already be patched, "
          "or differs from the expected commit. Not touching it.")
    sys.exit(1)

src = src.replace(old, new, 1)
open(path, "w").write(src)
print("patched OK")
