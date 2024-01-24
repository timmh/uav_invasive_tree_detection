import argparse
import os
import glob
import cv2
import shutil
import json
from sahi.slicing import slice_coco
from sahi.utils.coco import Coco


def main(args):

  shutil.rmtree(os.path.join(args.coco_annotation_dir, "yolo"), ignore_errors=True)
  shutil.rmtree(os.path.join(args.coco_annotation_dir, "sliced"), ignore_errors=True)


  with open(os.path.join(args.coco_annotation_dir, "instances_default.json")) as f:
    data = json.load(f)

  data["images"] = [{**image, "file_name": os.path.splitext(image["file_name"])[0] + ".tif"} for image in data["images"]]
  
  with open(os.path.join(args.coco_annotation_dir, "instances_default_tif.json"), "w") as f:
      data = json.dump(data, f)

  coco_dict, coco_path = slice_coco(
      coco_annotation_file_path=os.path.join(args.coco_annotation_dir, "instances_default_tif.json"),
      image_dir=args.coco_annotation_dir,
      output_coco_annotation_file_name=os.path.join(args.coco_annotation_dir, "instances_sliced"),
      output_dir=os.path.join(args.coco_annotation_dir, "sliced"),
      ignore_negative_samples=True,
      slice_height=640,
      slice_width=640,
      overlap_height_ratio=0.2,
      overlap_width_ratio=0.2,
  )


  # init Coco object
  coco = Coco.from_coco_dict_or_path(os.path.join(args.coco_annotation_dir, "instances_sliced_coco.json"), image_dir=os.path.join(args.coco_annotation_dir, "sliced/"))

  # export converted YoloV5 formatted dataset into given output_dir with a 85% train/15% val split
  coco.export_as_yolov5(
    output_dir=os.path.join(args.coco_annotation_dir, "yolo"),
    train_split_rate=0.8,
  )




  # fix class ids
  for subset in ["train", "val", "test"]:
    if not os.path.exists(os.path.join(args.coco_annotation_dir, "yolo", subset)):
      continue
    for fn in glob.glob(os.path.join(args.coco_annotation_dir, "yolo", subset, "*.txt")):

      img_fn = os.path.join(os.path.dirname(fn), os.path.splitext(os.path.basename(fn))[0] + ".png")
      img = cv2.imread(img_fn)
      if img is None:
        all_bg = True
      else:
        all_bg = (img == img[0, 0, :]).all()

      all_bg = False

      if all_bg:
        os.unlink(fn)
        os.unlink(img_fn)
      else:
        with open(fn) as f:
          d = f.readlines()
        d = [e.split(" ") for e in d]
        d = [[str(int(e[0]) - 1), *e[1:]] for e in d if len(e) > 0]
        with open(fn, "w") as f:
          f.writelines([" ".join(e) for e in d])


if __name__ == "__main__":
  argparser = argparse.ArgumentParser()
  argparser.add_argument("coco_annotation_dir", help="path to a directory with a COCO-formatted instances_default.json file")
  args = argparser.parse_args()
  main(args)