import asyncio
import base64
import websockets
import json, io
import cv2
import numpy as np
from PIL import Image


async def websocket_handler(websocket, path):

    print("WebSocket connection established.")
    try:
        async for message in websocket:

            print("Message received")

            data = json.loads(message)
            image_base64 = data["image"]

            compressed_image_bytes = base64.b64decode(image_base64)
            frame = decode_frame(compressed_image_bytes)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            print(frame.shape)

            copy_frame = frame.copy()

            """
            cv2.circle(frame, (x1, y1), r1, (255, 0, 0), 2)
            cv2.circle(frame, (x2, y2), r2, (255, 0, 0), 2)
            cv2.circle(frame, (x3, y3), r3, (255, 0, 0), 2)
            cv2.imwrite("output.jpg", frame)

            #cut frame to circle1 and circle2, it will be circular
            frame1 = cut_circle(frame, x1, y1, r1)
            frame2 = cut_circle(frame, x2, y2, r2)
            frame3 = cut_circle(frame, x3, y3, r3)
            

            response1 = await detect_target(frame1,1)
            response2 = await detect_target(frame2,2)
            response3 = await detect_target(frame3,3)
            data = enumarateData(response1, response2, response3)

            """

            response = await detect_target(frame)
            
            response = serialize_for_json(response)


            #await websocket.send(json.dumps({"status":True, "circle1": response1, "circle2": response2, "circle3": response3,  "data": data}))

            await websocket.send(json.dumps({"status":True, "detections": response}))

            print(response)



    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed: {str(e)}")
    except Exception as e:


        print(f"WebSocket Error: {str(e)}")


def serialize_for_json(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, dict):
        return {key: serialize_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [serialize_for_json(element) for element in data]
    else:
        return data


def decode_frame(compressed_image_bytes):
    image = Image.open(io.BytesIO(compressed_image_bytes))
    frame = np.array(image)
    return frame

def cut_circle(image, x, y, r):
    # Create a mask with the same dimensions as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Draw a filled circle on the mask
    cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)
    
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    # Optionally crop the image to the bounding box of the circle
    x_start = max(x - r, 0)
    y_start = max(y - r, 0)
    x_end = min(x + r, image.shape[1])
    y_end = min(y + r, image.shape[0])
    cropped_image = masked_image[y_start:y_end, x_start:x_end]
    
    return cropped_image


async def detect_target(frame):

    detections = []

    masked_img = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gürültüyü azaltmak için bulanıklaştır
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Hough Daire Dönüşümü ile daireleri tespit et
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=70,
                               param1=100, param2=80, minRadius=10, maxRadius=70)
    
    thresh = cv2.adaptiveThreshold(blurred, 255, 1, 1, 11, 2)

    # Eğer daireler bulunursa
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            # İçindeki üçgenleri bulma
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, (255, 255, 255), thickness=-1)

            masked_img = cv2.bitwise_and(gray, gray, mask=mask)
            edges = cv2.Canny(masked_img, 50, 150, apertureSize=3)
            
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=5, maxLineGap=10)

            if lines is not None:
                for i in range(len(lines)):
                    for j in range(i + 1, len(lines)):
                        line1 = lines[i][0]
                        line2 = lines[j][0]
                        if is_cross(line1, line2):
                            x1, y1, x2, y2 = line1
                            x3, y3, x4, y4 = line2
                            intersection = line_intersection((x1, y1, x2, y2), (x3, y3, x4, y4))
                            if intersection:
                                # if intersection is inside the circle, show circle
                                if (x - intersection[0]) ** 2 + (y - intersection[1]) ** 2 <= r ** 2:
                                    cv2.circle(frame, intersection, r, (0, 0, 255), 4)
                                    detections.append([intersection[0], intersection[1], r])

                                    
    cv2.imwrite("output.jpg", frame)           
    
    return detections
        
def is_cross(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A = (x1, y1)
    B = (x2, y2)
    C = (x3, y3)
    D = (x4, y4)
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)



def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    xdiff = (x1 - x2, x3 - x4)
    ydiff = (y1 - y2, y3 - y4)

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det((x1, y1), (x2, y2)), det((x3, y3), (x4, y4)))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)



"""
async def detect_yellow_dots(message, circle1, circle2):
    try:
        frame = decode_frame(message)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        print(frame.shape)
    


        output_frame = frame.copy()
     
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        lower_yellow = np.array([0, 130, 130])
        upper_yellow = np.array([60, 255, 255])

        lower_red = np.array([0, 0, 130])
        upper_red = np.array([60, 60, 255])


        mask = cv2.inRange(blurred_frame, lower_red, upper_red)
        mask = cv2.GaussianBlur(mask, (5, 5), 2)

        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=mask.shape[0] / 8, param1=100, param2=15, minRadius=5, maxRadius=50)

        circle_areas = [circle1, circle2]
        flags = [False, False]

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            circles = sorted(circles, key=lambda c: c[2], reverse=True)[:2]

            for i in range(len(circles)):
                x, y, r = circles[i]
                
                if check_bounds(x, y, circle_areas, i):
                    cv2.circle(output_frame, (x, y), r, (0, 255, 0), 2)
                    flags[i] = True
                else:
                    cv2.circle(output_frame, (x, y), r, (0, 0, 255), 2)
                    flags[i] = False

        cv2.circle(output_frame, (circle1[0], circle1[1]), circle1[2], (255, 0, 0), 2)
        cv2.circle(output_frame, (circle2[0], circle2[1]), circle2[2], (255, 0, 0), 2)

        cv2.imwrite("output.jpg", output_frame)

        data = enumarateData(flags[0], flags[1])
        return {"status": True , "circle1": flags[0], "circle2": flags[1], "data":data }

    except Exception as e:
        return {"status": False, "circle1": False, "circle2": False, "data": 0}



def check_bounds(x, y, circles, index):
    if (x - circles[0][0]) ** 2 + (y - circles[0][1]) ** 2 <= circles[0][2] ** 2:
        return True
    if (x - circles[1][0]) ** 2 + (y - circles[1][1]) ** 2 <= circles[1][2] ** 2:
        return True
    return False
"""


def enumarateData(circle1, circle2, circle3):
    c1 = 1 if circle1 else 0
    c2 = 1 if circle2 else 0
    c3 = 1 if circle3 else 0

    return [c1, c2, c3]
    

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        websockets.serve(websocket_handler, "0.0.0.0", 8765)
    )
    print("Server started...")
    loop.run_forever()
