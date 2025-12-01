package org.example.Fetch;

public class ProgressUpdate {
    private String phase;
    private String message;
    private double progress;
    private String algorithm;

    public ProgressUpdate(String algorithm, String phase, String message, double progress) {
        this.algorithm = algorithm;
        this.phase = phase;
        this.message = message;
        this.progress = progress;
    }

    // Getters
    public String getAlgorithm() { return algorithm; }
    public String getPhase() { return phase; }
    public String getMessage() { return message; }
    public double getProgress() { return progress; }
}