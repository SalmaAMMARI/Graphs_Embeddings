package org.example.Fetch;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.*;


import javax.swing.*;
import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamConstants;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import java.awt.*;
import java.io.*;
import java.util.*;
import java.util.List;
import java.awt.event.*;
import java.util.AbstractMap;

public class BipartiteGraphBuilder {
    public static void runAnalysisWithProgress(Graph graph, String algorithm) {
        System.out.println("\n" + "=".repeat(50));
        System.out.println("🔄 Starting " + algorithm + " Analysis");
        System.out.println("=".repeat(50));

        long startTime = System.currentTimeMillis();

        try {
            switch (algorithm.toLowerCase()) {
                case "node2vec":
                    runNode2VecAnalysis(graph);
                    break;
                case "deepwalk":
                    runDeepWalkAnalysis(graph);
                    break;
                case "graphsage":
                    runGraphSAGEAnalysis(graph);
                    break;
                case "gcn":
                    runGCNAnalysis(graph);
                    break;
                case "line":
                    runLINEAnalysis(graph);
                    break;
                default:
                    System.out.println("❌ Unknown algorithm: " + algorithm);
                    return;
            }

            long totalTime = System.currentTimeMillis() - startTime;
            System.out.printf("\n✅ %s analysis completed in %.2f minutes%n",
                    algorithm, totalTime / 60000.0);

        } catch (Exception e) {
            System.err.println("❌ Error during " + algorithm + " analysis: " + e.getMessage());
            e.printStackTrace();
        }
    }
    public static void runDeepWalkAnalysis(Graph graph) {
        System.out.println("\n=== DeepWalk Analysis ===");

        // Train DeepWalk
        DeepWalk deepwalk = new DeepWalk(
                graph,
                10,    // dimensions
                20,     // walkLength
                10,     // numWalks
                5,      // windowSize
                0.025,  // learningRate
                5       // epochs
        );

        deepwalk.train();

        // Get embeddings
        Map<String, double[]> embeddings = deepwalk.getEmbeddings();
        System.out.println("Generated embeddings for " + embeddings.size() + " nodes");

        // Test similarity
        System.out.println("\nMost similar authors to a sample author:");
        List<String> authors = new ArrayList<>(graph.getAuthorVertices());
        if (!authors.isEmpty()) {
            String sampleAuthor = authors.get(0);
            List<Map.Entry<String, Double>> similar = deepwalk.mostSimilar(sampleAuthor, 5);
            for (Map.Entry<String, Double> entry : similar) {
                System.out.printf("  %s: %.4f%n", entry.getKey(), entry.getValue());
            }
        }

        // Save embeddings
        deepwalk.saveEmbeddings("deepwalk_embeddings.txt");
        System.out.println("Embeddings saved to 'deepwalk_embeddings.txt'");
    }
    public static void runNode2VecAnalysis(Graph graph) {
        System.out.println("\n=== Node2Vec Analysis ===");

        // Train Node2Vec
        Node2Vec node2vec = new Node2Vec(
                graph,
                64,     // Fast, lower memory
                20,     // Short walks
                5,      // Few walks
                1.0,    // p
                1.0,    // q
                5,      // windowSize
                2       // Few epochs
        );

        node2vec.fit();

        // Get embeddings
        Map<String, double[]> embeddings = node2vec.getEmbeddings();
        System.out.println("Generated embeddings for " + embeddings.size() + " nodes");

        // Test similarity
        System.out.println("\nMost similar authors to a sample author:");
        List<String> authors = new ArrayList<>(graph.getAuthorVertices());
        if (!authors.isEmpty()) {
            String sampleAuthor = authors.get(0);
            List<Map.Entry<String, Double>> similar = node2vec.mostSimilar(sampleAuthor, 5);
            for (Map.Entry<String, Double> entry : similar) {
                System.out.printf("  %s: %.4f%n", entry.getKey(), entry.getValue());
            }
        }

        // Link prediction with progress tracking
        System.out.println("\n" + "=".repeat(60));
        System.out.println("🔗 STARTING LINK PREDICTION EVALUATION");
        System.out.println("=".repeat(60));

        LinkPredictor predictor = new LinkPredictor(embeddings, graph);

        // Evaluate link prediction
        predictor.evaluateLinkPrediction(10);

        // Show top predictions for sample nodes
        if (!authors.isEmpty()) {
            String sampleAuthor = authors.get(0);
            predictor.printTopPredictions(sampleAuthor, 5);
        }

        List<String> publications = new ArrayList<>(graph.getPublicationVertices());
        if (!publications.isEmpty()) {
            String samplePublication = publications.get(0);
            predictor.printTopPredictions(samplePublication, 5);
        }

        // Save embeddings
        node2vec.saveEmbeddings("node2vec_embeddings.txt");
        System.out.println("\nEmbeddings saved to 'node2vec_embeddings.txt'");
    }
    public static void runGraphSAGEAnalysis(Graph graph) {
        System.out.println("\n=== GraphSAGE Analysis ===");

        GraphSAGE graphsage = new GraphSAGE(
                graph,
                64,        // dimensions
                10,        // max neighbor samples
                0.01,      // learning rate
                10         // epochs
        );

        graphsage.fit();

        Map<String, double[]> embeddings = graphsage.getEmbeddings();
        System.out.println("Generated embeddings for " + embeddings.size() + " nodes");

        // Test similarity
        List<String> authors = new ArrayList<>(graph.getAuthorVertices());
        if (!authors.isEmpty()) {
            String sampleAuthor = authors.get(0);
            List<Map.Entry<String, Double>> similar = graphsage.mostSimilar(sampleAuthor, 5);
            System.out.println("\nMost similar authors to a sample author:");
            for (Map.Entry<String, Double> entry : similar) {
                System.out.printf("  %s: %.4f%n", entry.getKey(), entry.getValue());
            }
        }

        // Save
        graphsage.saveEmbeddings("graphsage_embeddings.txt");
        System.out.println("\nEmbeddings saved to 'graphsage_embeddings.txt'");
    }
    public static void runGCNAnalysis(Graph graph) {
        System.out.println("\n=== GCN Analysis ===");

        GCN gcn = new GCN(
                graph,
                64,        // dimensions
                0.01,      // learning rate
                10         // epochs
        );

        gcn.fit();

        Map<String, double[]> embeddings = gcn.getEmbeddings();
        System.out.println("Generated embeddings for " + embeddings.size() + " nodes");

        List<String> authors = new ArrayList<>(graph.getAuthorVertices());
        if (!authors.isEmpty()) {
            String sampleAuthor = authors.get(0);
            List<Map.Entry<String, Double>> similar = gcn.mostSimilar(sampleAuthor, 5);
            System.out.println("\nMost similar authors to a sample author:");
            for (Map.Entry<String, Double> entry : similar) {
                System.out.printf("  %s: %.4f%n", entry.getKey(), entry.getValue());
            }
        }

        gcn.saveEmbeddings("gcn_embeddings.txt");
        System.out.println("\nEmbeddings saved to 'gcn_embeddings.txt'");
    }
    public static void runLINEAnalysis(Graph graph) {
        System.out.println("\n=== LINE Analysis ===");

        LINE line = new LINE(
                graph,
                64,        // dimensions
                0.025,     // learning rate
                10,        // epochs
                5          // negative samples
        );

        line.fit();

        Map<String, double[]> embeddings = line.getEmbeddings();
        System.out.println("Generated embeddings for " + embeddings.size() + " nodes");

        List<String> authors = new ArrayList<>(graph.getAuthorVertices());
        if (!authors.isEmpty()) {
            String sampleAuthor = authors.get(0);
            List<Map.Entry<String, Double>> similar = line.mostSimilar(sampleAuthor, 5);
            System.out.println("\nMost similar authors to a sample author:");
            for (Map.Entry<String, Double> entry : similar) {
                System.out.printf("  %s: %.4f%n", entry.getKey(), entry.getValue());
            }
        }

        line.saveEmbeddings("line_embeddings.txt");
        System.out.println("\nEmbeddings saved to 'line_embeddings.txt'");
    }
    private final Map<String, Author> authors;
    private final Map<String, Publication> publications;
    private final Graph bipartiteGraph;

    public BipartiteGraphBuilder() {
        this.authors = new HashMap<>();
        this.publications = new HashMap<>();
        this.bipartiteGraph = new Graph();
    }

    public void parseXML(String xmlFilePath) {
        XMLInputFactory factory = XMLInputFactory.newInstance();
        try (InputStream inputStream = new FileInputStream(xmlFilePath)) {
            XMLStreamReader reader = factory.createXMLStreamReader(inputStream);

            String currentElement = "";
            Publication currentPublication = null;
            List<String> currentAuthors = new ArrayList<>();
            String currentPublicationType = null;

            while (reader.hasNext()) {
                int event = reader.next();

                switch (event) {
                    case XMLStreamConstants.START_ELEMENT:
                        currentElement = reader.getLocalName();

                        if ("incollection".equals(currentElement) ||
                                "book".equals(currentElement) ||
                                "phdthesis".equals(currentElement)) {

                            currentPublicationType = currentElement;
                            currentAuthors.clear();

                            String key = reader.getAttributeValue(null, "key");
                            String mdate = reader.getAttributeValue(null, "mdate");
                            String publtype = reader.getAttributeValue(null, "publtype");

                            currentPublication = new Publication(key, "", "", "", "", "", "", mdate, publtype, currentElement);
                        }
                        break;

                    case XMLStreamConstants.CHARACTERS:
                        String text = reader.getText().trim();
                        if (!text.isEmpty() && currentPublication != null) {
                            switch (currentElement) {
                                case "title":
                                    currentPublication.setTitle(text);
                                    break;
                                case "year":
                                    currentPublication.setYear(text);
                                    break;
                                case "pages":
                                    currentPublication.setPages(text);
                                    break;
                                case "booktitle":
                                    currentPublication.setBooktitle(text);
                                    break;
                                case "ee":
                                    currentPublication.setEe(text);
                                    break;
                                case "url":
                                    currentPublication.setUrl(text);
                                    break;
                                case "author":
                                case "editor":
                                    currentAuthors.add(text);
                                    break;
                            }
                        }
                        break;

                    case XMLStreamConstants.END_ELEMENT:
                        String elementName = reader.getLocalName();

                        if (("incollection".equals(elementName) ||
                                "book".equals(elementName) ||
                                "phdthesis".equals(elementName)) && currentPublication != null) {

                            // ADD ALL AUTHORS TO THE PUBLICATION
                            for (String authorName : currentAuthors) {
                                currentPublication.addAuthor(authorName);
                            }

                            publications.put(currentPublication.getKey(), currentPublication);
                            bipartiteGraph.addPublicationVertex(currentPublication.getKey(), currentPublication);

                            // Add ALL authors and create edges
                            for (String authorName : currentAuthors) {
                                Author author = authors.computeIfAbsent(authorName, Author::new);
                                author.addPublicationKey(currentPublication.getKey());

                                bipartiteGraph.addAuthorVertex(authorName, author);
                                bipartiteGraph.addEdge(authorName, currentPublication.getKey());
                            }

                            currentPublication = null;
                            currentAuthors.clear();
                        }
                        break;
                }
            }
            reader.close();

            printParsingStatistics();

        } catch (Exception e) {
            System.err.println("Error parsing XML: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void printParsingStatistics() {
        System.out.println("=== XML Parsing Statistics ===");
        System.out.println("Total authors found: " + authors.size());
        System.out.println("Total publications found: " + publications.size());

        // Calculate total edges from authors perspective
        int totalEdgesFromAuthors = authors.values().stream()
                .mapToInt(author -> author.getPublicationKeys().size())
                .sum();
        System.out.println("Total edges (from authors perspective): " + totalEdgesFromAuthors);

        // Calculate total edges from publications perspective
        int totalEdgesFromPublications = publications.values().stream()
                .mapToInt(pub -> pub.getAuthorNames().size())
                .sum();
        System.out.println("Total edges (from publications perspective): " + totalEdgesFromPublications);

        System.out.println("Edges match: " + (totalEdgesFromAuthors == totalEdgesFromPublications));

        // Print author statistics
        System.out.println("\n--- Author Statistics ---");
        long authorsWithNoPubs = authors.values().stream()
                .filter(author -> author.getPublicationKeys().isEmpty())
                .count();
        System.out.println("Authors with no publications: " + authorsWithNoPubs);

        // Print publication statistics
        System.out.println("\n--- Publication Statistics ---");
        long pubsWithNoAuthors = publications.values().stream()
                .filter(pub -> pub.getAuthorNames().isEmpty())
                .count();
        System.out.println("Publications with no authors: " + pubsWithNoAuthors);

        // Print edge distribution
        System.out.println("\n--- Edge Distribution ---");
        System.out.println("Average publications per author: " +
                String.format("%.2f", (double)totalEdgesFromAuthors / authors.size()));
        System.out.println("Average authors per publication: " +
                String.format("%.2f", (double)totalEdgesFromPublications / publications.size()));
    }

    // Getters
    public Map<String, Author> getAuthors() {
        return new HashMap<>(authors);
    }

    public Map<String, Publication> getPublications() {
        return new HashMap<>(publications);
    }

    public Graph getBipartiteGraph() {
        return bipartiteGraph;
    }

    public void printSampleData() {
        System.out.println("\n--- Sample Publications ---");
        int count = 0;
        for (Publication pub : publications.values()) {
            if (count >= 5) break;
            System.out.println("Publication: " + pub.getTitle());
            System.out.println("  Key: " + pub.getKey());
            System.out.println("  Authors: " + pub.getAuthorNames().size() + " - " + pub.getAuthorNames());
            System.out.println("  Year: " + pub.getYear());
            System.out.println();
            count++;
        }

        System.out.println("\n--- Sample Authors ---");
        count = 0;
        for (Author auth : authors.values()) {
            if (count >= 5) break;
            System.out.println("Author: " + auth.getName());
            System.out.println("  Publications: " + auth.getPublicationKeys().size());
            List<String> pubKeys = new ArrayList<>(auth.getPublicationKeys());
            if (!pubKeys.isEmpty()) {
                List<String> firstThree = pubKeys.subList(0, Math.min(3, pubKeys.size()));
                System.out.println("  Sample publications: " + firstThree);
            }
            System.out.println();
            count++;
        }
    }

    public void verifyGraphStructure() {
        System.out.println("\n=== Graph Structure Verification ===");

        Graph graph = getBipartiteGraph();

        System.out.println("Graph vertices: " + graph.getVertexCount());
        System.out.println("Author vertices: " + graph.getAuthorVertices().size());
        System.out.println("Publication vertices: " + graph.getPublicationVertices().size());
        System.out.println("Graph edges: " + graph.getEdgeCount());

        // Verify that edges are bidirectional
        boolean edgesCorrect = true;
        for (String author : graph.getAuthorVertices()) {
            for (String publication : graph.getNeighbors(author)) {
                if (!graph.getNeighbors(publication).contains(author)) {
                    System.out.println("ERROR: Edge not bidirectional: " + author + " -> " + publication);
                    edgesCorrect = false;
                }
            }
        }
        System.out.println("All edges bidirectional: " + edgesCorrect);

        // Verify bipartiteness
        boolean isBipartite = graph.isBipartite();
        System.out.println("Graph is bipartite: " + isBipartite);

        if (!isBipartite) {
            System.out.println("WARNING: Graph should be bipartite but isn't!");
        }
    }



    // NEW METHOD: Load embeddings from file
    private static Map<String, double[]> loadEmbeddings(String filename) {
        Map<String, double[]> embeddings = new HashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            int loaded = 0;

            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(" ");
                if (parts.length > 1) {
                    String nodeName = parts[0];
                    double[] vector = new double[parts.length - 1];

                    for (int i = 1; i < parts.length; i++) {
                        // Handle comma as decimal separator
                        String coord = parts[i].replace(",", ".");
                        vector[i-1] = Double.parseDouble(coord);
                    }

                    embeddings.put(nodeName, vector);
                    loaded++;

                    if (loaded % 1000 == 0) {
                        System.out.println("📥 Loaded " + loaded + " embeddings...");
                    }
                }
            }

            System.out.println("✅ Loaded " + loaded + " embeddings from " + filename);

        } catch (IOException e) {
            System.err.println("❌ Error loading embeddings: " + e.getMessage());
        }

        return embeddings;
    }

    // NEW METHOD: Calculate similarities between nodes
    private static void calculateSimilarities(Map<String, double[]> embeddings, Graph graph) {
        System.out.println("\n🔍 Calculating Cosine Similarities");
        System.out.println("=================================");

        // Get some sample nodes
        List<String> sampleNodes = new ArrayList<>();
        sampleNodes.addAll(new ArrayList<>(graph.getAuthorVertices())
                .subList(0, Math.min(3, graph.getAuthorVertices().size())));
        sampleNodes.addAll(new ArrayList<>(graph.getPublicationVertices())
                .subList(0, Math.min(3, graph.getPublicationVertices().size())));

        // Calculate similarity matrix
        for (int i = 0; i < sampleNodes.size(); i++) {
            for (int j = i + 1; j < sampleNodes.size(); j++) {
                String node1 = sampleNodes.get(i);
                String node2 = sampleNodes.get(j);

                if (embeddings.containsKey(node1) && embeddings.containsKey(node2)) {
                    double similarity = cosineSimilarity(embeddings.get(node1), embeddings.get(node2));
                    System.out.printf("Similarity between '%s' and '%s': %.4f\n",
                            node1, node2, similarity);
                }
            }
        }
    }

    // NEW METHOD: Cosine similarity calculation
    private static double cosineSimilarity(double[] vectorA, double[] vectorB) {
        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.pow(vectorA[i], 2);
            normB += Math.pow(vectorB[i], 2);
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}