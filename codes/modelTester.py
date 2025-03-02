# Load a model
model = YOLO("/home/adn/Desktop/best.pt")  # pretrained YOLO11n model
results = model(["/home/adn/Desktop/adna.jpg"])  # return


for i, result in enumerate(results):
    print(f"Results for Image {i+1}:")
    boxes = result.boxes  # Bounding box outputs
    if boxes is not None:
        for j, box in enumerate(boxes.xyxy):
            x_min, y_min, x_max, y_max = box.tolist()
            print(f"Object {j+1}: x_min={int(x_min)}, y_min={int(y_min)}, x_max={int(x_max)}, y_max={int(y_max)}")
   
    result.show()  # Display image with detections
    result.save(filename=f"result_{i+1}.jpg")  # Save to disk