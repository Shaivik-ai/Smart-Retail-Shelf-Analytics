"""Step 3 verification: run vision_engine over the shelf video with keyboard controls."""

import argparse
import json
import sys

import cv2

from vision_engine import ShelfVisionEngine

ZONE_KEYS = ["Sprite_Zone", "Coca_Zone", "Fanta_Zone"]


def parse_args():
    p = argparse.ArgumentParser(description="Step 3: CV + spatial shelf logic")
    p.add_argument("--video", default="Dataset/Bevrage_Shelf.mp4")
    p.add_argument("--weights", default="best.pt")
    p.add_argument("--save-output", default=None, help="path to write annotated video")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    engine = ShelfVisionEngine(args.weights, device=args.device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open {args.video}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if args.save_output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_output, fourcc, fps, (w, h))

    print("Controls: 1/2/3 = remove bottle | r = reset | m = toggle misplaced | s = snapshot+JSON | q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop for interactive testing
            continue

        annotated, telemetry = engine.process_frame(frame)

        if writer:
            writer.write(annotated)

        cv2.imshow("Step 3 - Shelf Vision", annotated)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("1"), ord("2"), ord("3")):
            engine.simulated_removed[ZONE_KEYS[key - ord("1")]] += 1
        elif key == ord("r"):
            engine.simulated_removed = {z: 0 for z in ZONE_KEYS}
            engine.simulate_misplaced = False
        elif key == ord("m"):
            engine.simulate_misplaced = not engine.simulate_misplaced
        elif key == ord("s"):
            cv2.imwrite("step3_verification.jpg", annotated)
            print(json.dumps(telemetry, indent=2))

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())