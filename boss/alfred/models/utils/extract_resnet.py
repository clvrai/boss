import os
import sys

sys.path.append(os.path.join(os.environ["ALFRED_ROOT"]))
sys.path.append(os.path.join(os.environ["ALFRED_ROOT"], "models"))

import torch
import os
from PIL import Image
from nn.resnet import Resnet
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument("--data", help="data folder", default="data/2.1.0")
    parser.add_argument("--batch", help="batch size", default=256, type=int)
    parser.add_argument("--gpu", help="use gpu", action="store_true")
    parser.add_argument(
        "--skip_existing",
        help="skip folders that already have feats",
        action="store_true",
    )
    parser.add_argument(
        "--visual_model",
        default="resnet18",
        help="model type: maskrcnn or resnet18 or resnet50 or mvp",
        choices=["maskrcnn", "resnet18", "resnet50", "mvp"],
    )
    parser.add_argument("--filename", help="filename of feat", default="feat_conv.pt")
    parser.add_argument(
        "--img_folder", help="folder containing raw images", default="raw_images"
    )

    # parser
    args = parser.parse_args()

    # load resnet model
    extractor = Resnet(args, eval=True)
    skipped = []

    for root, dirs, files in os.walk(args.data):
        if os.path.basename(root) == args.img_folder:
            fimages = sorted(
                [
                    os.path.join(root, f)
                    for f in files
                    if (f.endswith(".png") or (f.endswith(".jpg")))
                ]
            )
            if len(fimages) > 0:
                if args.skip_existing and os.path.isfile(
                    os.path.join(root.replace(args.img_folder, ""), args.filename)
                ):
                    continue
                try:
                    print("{}".format(root))
                    image_loader = (
                        Image.open if isinstance(fimages[0], str) else Image.fromarray
                    )
                    images = [image_loader(f) for f in fimages]
                    feat = extractor.featurize(images, batch=args.batch)
                    torch.save(
                        feat.cpu(),
                        os.path.join(root.replace(args.img_folder, ""), args.filename),
                    )
                except Exception as e:
                    try:
                        first_images = [
                            image_loader(f) for f in fimages[: len(fimages) // 2]
                        ]
                        feat = extractor.featurize(first_images, batch=args.batch)
                        for img in first_images:
                            img.close()
                        second_images = [
                            image_loader(f) for f in fimages[len(fimages) // 2 :]
                        ]
                        feat2 = extractor.featurize(second_images, batch=args.batch)
                        torch.save(
                            torch.cat((feat, feat2), dim=0).cpu(),
                            os.path.join(
                                root.replace(args.img_folder, ""), args.filename
                            ),
                        )
                    except Exception as e2:
                        print(e)
                        print(e2)
                        print("Skipping " + root)
                        skipped.append(root)

            else:
                print("empty; skipping {}".format(root))

    print("Skipped:")
    print(skipped)
