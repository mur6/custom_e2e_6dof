REMOTE_PATH=/home/ubuntu/workspace/2024/6DoF/E2E_Object_Pose_Estimator
scp *.py ml-ec2:${REMOTE_PATH}
scp -r utils ml-ec2:${REMOTE_PATH}/
scp -r rob599 ml-ec2:${REMOTE_PATH}/
scp requirements.txt ml-ec2:${REMOTE_PATH}
