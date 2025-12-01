package org.example.Fetch;

import java.util.List;
import java.util.Map;

public class AnalysisResult {
    private String algorithm;
    private long executionTime;
    private int embeddingsGenerated;
    private List<Map.Entry<String, Double>> similarityResults;
    private Map<String, Double> linkPredictionMetrics;
    private String status;

    public AnalysisResult(String algorithm) {
        this.algorithm = algorithm;
        this.status = "Pending";
    }

    // Getters and setters
    public String getAlgorithm() { return algorithm; }
    public void setAlgorithm(String algorithm) { this.algorithm = algorithm; }

    public long getExecutionTime() { return executionTime; }
    public void setExecutionTime(long executionTime) { this.executionTime = executionTime; }

    public int getEmbeddingsGenerated() { return embeddingsGenerated; }
    public void setEmbeddingsGenerated(int embeddingsGenerated) { this.embeddingsGenerated = embeddingsGenerated; }

    public List<Map.Entry<String, Double>> getSimilarityResults() { return similarityResults; }
    public void setSimilarityResults(List<Map.Entry<String, Double>> similarityResults) { this.similarityResults = similarityResults; }

    public Map<String, Double> getLinkPredictionMetrics() { return linkPredictionMetrics; }
    public void setLinkPredictionMetrics(Map<String, Double> linkPredictionMetrics) { this.linkPredictionMetrics = linkPredictionMetrics; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}