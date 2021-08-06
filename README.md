## AI System for Autonomous Driving Real-time Detection

Computer vision has revolutionized the self-driving industry. Object detection and tracking are essential tasks that allow vehicles to identify obstacles in its course and take action. We hope to develop an AI system that is useful in self-driving vehicles. 

### Goals
+ Develop an AI system to recognize obstacles (at least pedestrians & cars) 
+ Determine distances to the identified objects and give collision warnings when needed

### Detection
Detection is performed with [CenterNet model](https://github.com/xingyizhou/CenterNet), pretrained on COCO dataset. 

<img src="https://github.com/zhangyuqing/autonomous_driving/blob/main/examples/det.gif" alt="Scene 1" width="600"/>
