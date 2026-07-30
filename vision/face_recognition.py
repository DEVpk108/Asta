import cv2
import pickle
import face_recognition
import os



class FaceRecognizer:
    def face_reco(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        
        
        if not os.path.exists("assets/face_encodings.pkl"):
            print("Error: face_encodings.pkl file not found!")
            cap.release()
            return

        with open("assets/face_encodings.pkl", "rb") as f:
            known_encodings = pickle.load(f)
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame,1)
        
            if ret:
                #cv2.imshow("Video Feed", frame)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                
            try:        
                if True in matches:
                    return True
            except:
                pass
            
            #key = cv2.waitKey(1)
            #if key == ord("q"):
                #break
         
    
        cap.release()
        cv2.destroyAllWindows()
    

  
if __name__ == '__main__':
    obj = FaceRecognizer()
    obj.face_reco()