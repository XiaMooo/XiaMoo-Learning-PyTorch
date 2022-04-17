import cv2
import numpy as np
import torch
from MNIST_TRAIN import ConvNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img = np.zeros((900, 960, 3), dtype=np.uint8)
img.fill(255)
norm = np.zeros((28, 28), dtype=np.float32)

window_name = "XiaMoo MNIST Viewer"
square_size = 22
pen_size = 35
drawing = False
mode = True

ix, iy = -1, -1
px, py = -1, -1
reset = True


def draw(event, x, y, flags, param):
    global ix, iy, drawing, px, py, reset
    if event == cv2.EVENT_LBUTTONDOWN and reset:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), pen_size, (255, 255, 255), -1)
            px, py = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_RBUTTONDOWN and reset:
        drawing = False
        reset = False
        pool()
        predict(np.array([[norm]]))


def pad_reset():
    img[650:850].fill(255)
    for i in range(28):
        for j in range(28):
            x0, y0 = 172 + i * square_size, j * square_size
            x1, y1 = x0 + square_size - 2, y0 + square_size - 2
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 0), -1)


def pool():
    for i in range(28):
        for j in range(28):
            x0, y0 = 172 + i * square_size, j * square_size
            x1, y1 = x0 + square_size - 2, y0 + square_size - 2
            square = img[y0:y1, x0:x1]
            # print(square.shape)
            mean = np.mean(square)
            img[y0:y1 + 1, x0:x1 + 1].fill(mean)
            norm[j][i] = (mean / 255.0) ** 2


def predict(inputs):
    global reset
    inputs = torch.from_numpy(inputs).to(device)
    outputs = model(inputs)
    pred = [i for i in outputs[0].cpu().numpy() + 2]
    print(num := pred.index(max(pred)))
    # print("[ " + ", ".join([f"{i:+.2f}" for i in pred]), end="\n\n")
    cv2.putText(img, str(num), (420, 800), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 6, (32, 128, 255), 10)
    reset = True


if __name__ == '__main__':
    model = torch.load("./MNIST.pth")
    model.eval()
    model.to(device)
    cv2.namedWindow(window_name)
    pad_reset()
    cv2.setMouseCallback(window_name, draw)
    with torch.no_grad():
        while True:
            cv2.imshow(window_name, img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == ord('c') or k == ord('r'):
                pad_reset()
                reset = True

    cv2.destroyAllWindows()
