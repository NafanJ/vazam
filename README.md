# Voice Recognition API

This project provides a REST API for voice recognition using a Python script and Java Spring Boot.

## Prerequisites

- Java 11
- Maven
- Python 3
- Required Python libraries: librosa, numpy, faiss

## Setup

1. Clone the repository.
2. Navigate to the project directory.
3. ./mvnw spring-boot:run
4. curl -v -F "file=@src/main/resources/audio/seth_mcfarlane_test.mp3" http://localhost:8080/recognize

### Install Python Dependencies

```sh
pip install librosa numpy faiss-cpu
