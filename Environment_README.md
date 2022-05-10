# Set up Container

- Set-up compute environment with containers:
    - Install [Singularity](https://sylabs.io/docs/)
      - Code was tested with Singularity 2.6.1 on a RedHat linux host
      - Based on a docker image, so likely docker compatible
    - Ensure CUDA 10.1+ compatible GPU (e.g. V100) and drivers are present.  
      - Note: Our tests were performed using nodes containing a single V100 GPU and 374 GB of memory. 
    - Pull image file from [Docker Hub](https://hub.docker.com/layers/srajaram/lab_img/v2.14/images/sha256-f307fdf489b509740758813abef603f931993ce9ea1979eb9d980853a4b52)
      - `singularity pull docker://srajaram/lab_img:v2.14`
    - Test GPU (make sure singularity image is in current working directory):
      -  `singularity exec --nv --cleanenv ./lab_img-v2.14.simg nvidia-smi`
    - Initialize Singularity image (make sure singularity image is in current working directory):
      - `singularity shell --nv -B /etc/machine-id -B /home2 ./lab_img-v2.14.simg`
    - Note: directories in above line (ex. home2) should be changed to reflect where your container has been stored 



# Running Code

Note: Be sure to change paths in [Parameters/Project_Paths.yaml](Parameters/Project_Paths.yaml) as indicated in the [Data download instructions](Data_Instructions.md) to make sure the code can find the data.

All instructions provided assume you have:

1. Cloned this repo to somewhere on your computer/cluster.
2. Fired up a singularity image as indicated above
3. Navigated to the root directory where you have saved the code before launching any commands.



