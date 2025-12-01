package org.example.Fetch;

import java.util.List;
import java.util.Map;

public class GraphStats {
    private int totalAuthors;
    private int totalPublications;
    private int totalEdges;
    private double avgPublicationsPerAuthor;
    private double avgAuthorsPerPublication;
    private List<String> highDegreeVertices;
    private int maximumMatchingSize;
    private int authorIndependentSetSize;
    private int publicationIndependentSetSize;

    public GraphStats() {}

    // Getters and setters
    public int getTotalAuthors() { return totalAuthors; }
    public void setTotalAuthors(int totalAuthors) { this.totalAuthors = totalAuthors; }

    public int getTotalPublications() { return totalPublications; }
    public void setTotalPublications(int totalPublications) { this.totalPublications = totalPublications; }

    public int getTotalEdges() { return totalEdges; }
    public void setTotalEdges(int totalEdges) { this.totalEdges = totalEdges; }

    public double getAvgPublicationsPerAuthor() { return avgPublicationsPerAuthor; }
    public void setAvgPublicationsPerAuthor(double avgPublicationsPerAuthor) { this.avgPublicationsPerAuthor = avgPublicationsPerAuthor; }

    public double getAvgAuthorsPerPublication() { return avgAuthorsPerPublication; }
    public void setAvgAuthorsPerPublication(double avgAuthorsPerPublication) { this.avgAuthorsPerPublication = avgAuthorsPerPublication; }

    public List<String> getHighDegreeVertices() { return highDegreeVertices; }
    public void setHighDegreeVertices(List<String> highDegreeVertices) { this.highDegreeVertices = highDegreeVertices; }

    public int getMaximumMatchingSize() { return maximumMatchingSize; }
    public void setMaximumMatchingSize(int maximumMatchingSize) { this.maximumMatchingSize = maximumMatchingSize; }

    public int getAuthorIndependentSetSize() { return authorIndependentSetSize; }
    public void setAuthorIndependentSetSize(int authorIndependentSetSize) { this.authorIndependentSetSize = authorIndependentSetSize; }

    public int getPublicationIndependentSetSize() { return publicationIndependentSetSize; }
    public void setPublicationIndependentSetSize(int publicationIndependentSetSize) { this.publicationIndependentSetSize = publicationIndependentSetSize; }
}