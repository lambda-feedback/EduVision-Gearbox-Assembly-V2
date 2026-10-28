# Python Evaluation Function

This repository contains the boilerplate code needed to create a containerized evaluation function written in Python.

## Deployment
[![Create Release Request](https://img.shields.io/badge/Create%20Release%20Request-blue?style=for-the-badge)](https://github.com/lambda-feedback/{REPO_NAME_HERE}/issues/new?template=release-request.yml)
To deploy to production, update the README button above to point to the correct repository.

## Quickstart

This chapter helps you to quickly set up a new Python evaluation function using this template repository.

> [!NOTE]
> After setting up the evaluation function, delete this chapter from the `README.md` file, and add your own documentation.

#### 1. Create a new repository

- In GitHub, choose `Use this template` > `Create a new repository` in the repository toolbar.

- Choose the owner, and pick a name for the new repository.

  > [!IMPORTANT]
  > If you want to deploy the evaluation function to Lambda Feedback, make sure to choose the Lambda Feedback organization as the owner.

- Set the visibility to `Public` or `Private`.

  > [!IMPORTANT]
  > If you want to use GitHub [deployment protection rules](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment#deployment-protection-rules), make sure to set the visibility to `Public`.

- Click on `Create repository`.

#### 2. Clone the new repository

Clone the new repository to your local machine using the following command:

```bash
git clone <repository-url>
```

#### 3. Configure the evaluation function

When deploying to Lambda Feedback, set the evaluation function name in the `config.json` file. Read the [Deploy to Lambda Feedback](#deploy-to-lambda-feedback) section for more information.

#### 4. Develop the evaluation function

You're ready to start developing your evaluation function. Head over to the [Development](#development) section to learn more.

#### 5. Update the README

In the `README.md` file, change the title and description so it fits the purpose of your evaluation function.

Also, don't forget to delete the Quickstart chapter from the `README.md` file after you've completed these steps.

## Usage

You can run the evaluation function either using [the pre-built Docker image](#run-the-docker-image) or build and run [the binary executable](#build-and-run-the-binary).

### Run the Docker Image

The pre-built Docker image comes with [Shimmy](https://github.com/lambda-feedback/shimmy) installed.

> [!TIP]
> Shimmy is a small application that listens for incoming HTTP requests, validates the incoming data and forwards it to the underlying evaluation function. Learn more about Shimmy in the [Documentation](https://github.com/lambda-feedback/shimmy).

The pre-built Docker image is available on the GitHub Container Registry. You can run the image using the following command:

```bash
docker run -p 8080:8080 ghcr.io/lambda-feedback/evaluation-function-boilerplate-python:latest
```

### Run the Script

You can choose between running the Python evaluation function itself, ore using Shimmy to run the function.

**Raw Mode**

Use the following command to run the evaluation function directly:

```bash
python -m evaluation_function.main
```

This will run the evaluation function using the input data from `request.json` and write the output to `response.json`.

**Shimmy**

To have a more user-friendly experience, you can use [Shimmy](https://github.com/lambda-feedback/shimmy) to run the evaluation function.

To run the evaluation function using Shimmy, use the following command:

```bash
shimmy -c "python" -a "-m" -a "evaluation_function.main" -i ipc
```

## Development

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Python](https://www.python.org)

### Repository Structure

```bash
evaluation_function/main.py             # evaluation function entrypoint
evaluation_function/evaluation.py       # evaluation function implementation
evaluation_function/evaluation_test.py  # evaluation function tests
evaluation_function/preview.py          # evaluation function preview
evaluation_function/preview_test.py     # evaluation function preview tests

config.json                             # evaluation function deployment configuration file
```

### Development Workflow

In its most basic form, the development workflow consists of writing the evaluation function in the `evaluation_function/evaluation.py` file and testing it locally. As long as the evaluation function adheres to the Evaluation Function API, a development workflow which incorporates using Shimmy is not necessary.

Testing the evaluation function can be done by running the `dev.py` script using the Python interpreter like so:

```bash
python -m evaluation_function.dev <response> <answer>
```

> [!NOTE]
> Specify the `response` and `answer` as command-line arguments.

### Building the Docker Image

To build the Docker image, run the following command:

```bash
docker build -t my-python-evaluation-function .
```

### Running the Docker Image

To run the Docker image, use the following command:

```bash
docker run -it --rm -p 8080:8080 my-python-evaluation-function
```

This will start the evaluation function and expose it on port `8080`.

## Deployment

This section guides you through the deployment process of the evaluation function. If you want to deploy the evaluation function to Lambda Feedback, follow the steps in the [Lambda Feedback](#deploy-to-lambda-feedback) section. Otherwise, you can deploy the evaluation function to other platforms using the [Other Platforms](#deploy-to-other-platforms) section.

### Deploy to Lambda Feedback

Deploying the evaluation function to Lambda Feedback is simple and straightforward, as long as the repository is within the [Lambda Feedback organization](https://github.com/lambda-feedback).

After configuring the repository, a [GitHub Actions workflow](.github/workflows/deploy.yml) will automatically build and deploy the evaluation function to Lambda Feedback as soon as changes are pushed to the main branch of the repository.

**Configuration**

The deployment configuration is stored in the `config.json` file. Choose a unique name for the evaluation function and set the `EvaluationFunctionName` field in [`config.json`](config.json).

> [!IMPORTANT]
> The evaluation function name must be unique within the Lambda Feedback organization, and must be in `lowerCamelCase`. You can find a example configuration below:

```json
{
  "EvaluationFunctionName": "compareStringsWithPython"
}
```

### Deploy to other Platforms

If you want to deploy the evaluation function to other platforms, you can use the Docker image to deploy the evaluation function.

Please refer to the deployment documentation of the platform you want to deploy the evaluation function to.

If you need help with the deployment, feel free to reach out to the Lambda Feedback team by creating an issue in the template repository.

## FAQ

### Pull Changes from the Template Repository

If you want to pull changes from the template repository to your repository, follow these steps:

1. Add the template repository as a remote:

```bash
git remote add template https://github.com/lambda-feedback/evaluation-function-boilerplate-python.git
```

2. Fetch changes from all remotes:

```bash
git fetch --all
```

3. Merge changes from the template repository:

```bash
git merge template/main --allow-unrelated-histories
```

> [!WARNING]
> Make sure to resolve any conflicts and keep the changes you want to keep.

## Troubleshooting

### Containerized Evaluation Function Fails to Start

If your evaluation function is working fine when run locally, but not when containerized, there is much more to consider. Here are some common issues and solution approaches:

**Run-time dependencies**

Make sure that all run-time dependencies are installed in the Docker image.

- Python packages: Make sure to add the dependency to the `pyproject.toml` file, and run `poetry install` in the Dockerfile.
- System packages: If you need to install system packages, add the installation command to the Dockerfile.
- ML models: If your evaluation function depends on ML models, make sure to include them in the Docker image.
- Data files: If your evaluation function depends on data files, make sure to include them in the Docker image.

**Architecture**

Some package may not be compatible with the architecture of the Docker image. Make sure to use the correct platform when building and running the Docker image.

E.g. to build a Docker image for the `linux/x86_64` platform, use the following command:

```bash
docker build --platform=linux/x86_64 .
```

**Verify Standalone Execution**

If requests are timing out, it might be due to the evaluation function not being able to run. Make sure that the evaluation function can be run as a standalone script. This will help you to identify issues that are specific to the containerized environment.

To run just the evaluation function as a standalone script, without using Shimmy, use the following command:

```bash
docker run -it --rm my-python-evaluation-function python -m evaluation_function.main
```

If the command starts without any errors, the evaluation function is working correctly. If not, you will see the error message in the console.
