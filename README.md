# GPU Examples and Exercises for Parallel Computing Course (PAC) at FHNW 2026

Welcome to the official GitHub repository containing all the examples and exercises related to the GPU section of the Parallel Computing (PAC) course at the Fachhochschule Nordwestschweiz (FHNW). This repository is designed to help students manage and execute their code using Visual Studio Code as their IDE.

## Features

- **Comprehensive Examples and Exercises**: Each CUDA file contains relevant examples and exercises as per the PAC curriculum focused on GPU computing.
- **IDE Support**: Configured to use Visual Studio Code, enhancing the development experience with features tailored for GPU programming.
- **Dynamic Compilation and Execution**: Compile, run, debug, and profile tasks are set up to operate on the "current file" opened in VS Code, making it straightforward to work on individual exercises and examples.
- **Weekly Updates**: New exercises will be added weekly, corresponding with the course schedule. Ensure you pull the latest changes to stay up to date.

## Setup
1. **Install VS Code**: Download and instal [VS Code](https://code.visualstudio.com/download)

2. **Upload your SSH key**: See slides

3. **Access to your GPU**: Make a remote connection to the IP of our PAC GPU server inside the Remote Dev plugin. Use your SSH key to login. Make sure the file permissions of the SSH keys are read-only for your user (400 in Linux).
    ```sh
    ssh -i id_rsa_pac vorname.nachname@10.35.149.100
    ```

4. **Install nsight VS code plugin**: Install on the remote VS code instance the "Nsight" plugin from Nvidia. This enables GPU debugging inside VS Code.

5. **Install Nsight profiling Tools**: Install [Nsight Systems](https://developer.nvidia.com/nsight-systems/get-started) and [Nsight Compute](https://developer.nvidia.com/tools-overview/nsight-compute/get-started) locally on **your machine**. This enables you to open the profiling files generated on the GPU enabled remote system.


## Getting Started

To get started with the repository, clone it to your GPU enabled machine and open the directory using VS Code.

## How to Use

- Open a file: Navigate to the file you wish to work on.
- Compile/Run: Use the pre-configured tasks in VS Code (accessible via Terminal -> Run Task in VS Code).
- Debug/Profile: Launch the debugging or profiling tools through the Run menu in VS Code.

Ensure that your development environment is set up with all the necessary dependencies for GPU computing, including the correct drivers and SDKs.
Install the Nvidia VS Code extension "nsight"

## Contributing

Feel free to fork this repository and submit pull requests to contribute to the exercises. For major changes, please open an issue first to discuss what you would like to change.


## Support

If you encounter any problems, please ask in classes or write in the MS Teams Forum.

Happy Coding!
