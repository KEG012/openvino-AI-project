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

# Software flow chart

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
