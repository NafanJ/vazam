package com.example.voicerecognition;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.util.logging.Logger;

@RestController
public class VoiceRecognitionController {

    private static final Logger logger = Logger.getLogger(VoiceRecognitionController.class.getName());

    @Autowired
    private VoiceRecognitionService voiceRecognitionService;

    @PostMapping("/recognize")
    public String recognizeVoice(@RequestParam("file") MultipartFile file) {
        logger.info("Endpoint /recognize called");
        return voiceRecognitionService.recognizeVoice(file);
    }
}
