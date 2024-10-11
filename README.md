# openvino-AI-project
Gihub got openvino AI project(Human chasing mobility)


# Project Gantt Chart

```mermaid
gantt
    title Project Schedule
    dateFormat  MM/DD
    section Market Research
    Pre-research        :active,   pre-research, 09/25, 1d
    section AI Model
    Human Detection     :active,   human-detection, 09/25, 3d
    Hand Detection      :          hand-detection, 09/27, 3d
    Hand Gesture Classification:   hand-gesture-classification, 09/30, 3d
    Mono-eye Distance Measurement: mono-eye-distance, 09/25, 6d
    Color Detection     :          color-detection, 09/25, 6d
    Segmentation Model  :          segmentation, 09/25, 10d
    Model Synthesis (Ensemble) :    model-synthesis, 09/30, 9d
    Model Testing       :          model-testing, 09/27, 14d
    section Hardware
    Hardware Setup      :          hardware-setup, 10/08, 3d
    Vehicle Algorithm Development:  vehicle-algorithm, 10/10, 7d
    Embedded Software Development: embedded-software, 10/10, 7d
    section Total Inspection
    Debugging           :          debugging, 10/16, 6d
    Testing             :          testing, 10/16, 7d
    Rehearsal           :          rehearsal, 10/21, 2d
```

# Software Flow Chart

```mermaid
flowchart TD
    A[Start: Chasing Car Activated] --> B[Depth Camera Input]
    B --> C[Person Detection]
    C --> D[Pose Estimation: Shoulders, Hips, Elbows, Head]
    D --> E[Create Landmark Based on Body Features]
    E --> F[Identify Person]
    F --> G[Is Person Showing Palm Hand Gesture?]
    G -- Yes --> H[Detect Specific Person Only]
    H --> I[Follow Detected Person]
    G -- No --> B
    I --> J[Hand Gesture Recognition]
    J --> K{Gesture Command}
    K -- One Finger --> L[Move Closer]
    K -- Two Fingers --> M[Move Backward]
    K -- Three Fingers --> N[Rotate Right]
    K -- Four Fingers --> O[Rotate Left]
    K -- Fist --> P[Stop Following]
    P --> End[End Process]
    
    I --> Q[Has Person Disappeared from View?]
    Q -- Yes --> R[Move Forward 1 Meter]
    R --> S{Was Person on Left or Right of Screen?}
    S -- Left --> T[Turn Left]
    S -- Right --> U[Turn Right]
    T --> V[Search for Person for 10 Seconds]
    U --> V[Search for Person for 10 Seconds]
    V --> W{Is Person Found?}
    W -- Yes --> I
    W -- No --> X[Stop and End Process]
    X --> End
    Q -- No --> J
    
    N --> Y[Rotate Right for 5 Seconds] --> Z[Return to Center]
    O --> AA[Rotate Left for 5 Seconds] --> Z[Return to Center]
    Z --> AB[Search for Specific Person]
    AB --> V
```



# HLD1 (젯슨나노에서 모델을 돌리기 어려운 경우)
<img src="./HLD1.png" alt="이미지 설명" width="500" height="400"/>

# HLD2 (젯슨나노에서 모델을 돌릴 수 있을 경우)
<img src="./HLD2.png" alt="이미지 설명" width="300" height="300"/>




# finished job
1. ROS를 통한 turtle bot control Check
2. jetson nano와 intel realsense connection Check
3. 라즈베리에서 intel realsense 사용 불가 Check
4. mediapipe 를 사용한 hand gesture recognition Check


# current job
은찬, 태섭 : 통신 속도 정상화, turtlebot 제어 코드 refactoring, frame 처리 및 동작 제어 threading ,제스쳐 명령 수행<br>
동현, 의근 : 
