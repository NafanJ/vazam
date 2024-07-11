package com.example.voicerecognition;

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.logging.Logger;

@Service
public class VoiceRecognitionService {

    private static final Logger logger = Logger.getLogger(VoiceRecognitionService.class.getName());
    private static final String PYTHON_SCRIPT_PATH = "scripts/voice_actor_recognition.py";

    public String recognizeVoice(MultipartFile file) {
        logger.info("Received file: " + file.getOriginalFilename());

        // Save the file to a temporary location
        File tempFile = convertMultipartFileToFile(file);
        if (tempFile == null) {
            logger.severe("File conversion failed.");
            return "File conversion failed.";
        }

        logger.info("File saved to temporary location: " + tempFile.getAbsolutePath());

        // Call the Python script
        try {
            ProcessBuilder pb = new ProcessBuilder("python3", PYTHON_SCRIPT_PATH, tempFile.getAbsolutePath());
            pb.redirectErrorStream(true);
            Process process = pb.start();
            
            int exitCode = process.waitFor();
            logger.info("Python script exit code: " + exitCode);

            String output = new String(Files.readAllBytes(Paths.get("output.txt")));
            logger.info("Python script output: " + output);
            return output;
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            logger.severe("Error during voice recognition: " + e.getMessage());
            return "Error during voice recognition.";
        } finally {
            // Clean up the temporary file
            if (tempFile.exists()) {
                tempFile.delete();
            }
        }
    }

    private File convertMultipartFileToFile(MultipartFile file) {
        try {
            File convFile = new File(file.getOriginalFilename());
            FileOutputStream fos = new FileOutputStream(convFile);
            fos.write(file.getBytes());
            fos.close();
            return convFile;
        } catch (IOException e) {
            e.printStackTrace();
            logger.severe("Error converting file: " + e.getMessage());
            return null;
        }
    }
}
