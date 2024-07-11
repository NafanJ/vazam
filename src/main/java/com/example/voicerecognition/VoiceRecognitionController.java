package com.example.voicerecognition;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

@RestController
public class VoiceRecognitionController {

    @Autowired
    private VoiceRecognitionService voiceRecognitionService;

    @PostMapping("/recognize")
    public String recognizeVoice(@RequestParam("file") MultipartFile file) {
        return voiceRecognitionService.recognizeVoice(file);
    }
}
