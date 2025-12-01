package org.example.Fetch;

import java.util.*;
import java.io.*;

public class DeepWalk {
    private final Graph graph;
    private final int dimensions;
    private final int walkLength;
    private final int numWalks;
    private final int windowSize;
    private final double learningRate;
    private final int epochs;

    private Map<String, double[]> embeddings;
    private Random random;
    private long startTime;

    public DeepWalk(Graph graph, int dimensions, int walkLength, int numWalks,
                    int windowSize, double learningRate, int epochs) {
        this.graph = graph;
        this.dimensions = dimensions;
        this.walkLength = walkLength;
        this.numWalks = numWalks;
        this.windowSize = windowSize;
        this.learningRate = learningRate;
        this.epochs = epochs;
        this.random = new Random(42);
        this.embeddings = new HashMap<>();
    }

    // Generate random walks from each vertex with progress tracking
    private List<List<String>> generateRandomWalks() {
        System.out.println("🎯 Starting random walk generation...");

        // Get all vertices from both authors and publications
        Set<String> allVertices = new HashSet<>(graph.getAuthorVertices());
        allVertices.addAll(graph.getPublicationVertices());

        System.out.println("   Total walks to generate: " + (numWalks * allVertices.size()));

        List<List<String>> allWalks = new ArrayList<>();
        List<String> vertexList = new ArrayList<>(allVertices);
        int totalWalks = numWalks * vertexList.size();
        int walksGenerated = 0;
        long lastPrintTime = System.currentTimeMillis();

        for (int walkNum = 0; walkNum < numWalks; walkNum++) {
            System.out.printf("   🔄 Walk batch %d/%d%n", walkNum + 1, numWalks);

            for (int vertexIdx = 0; vertexIdx < vertexList.size(); vertexIdx++) {
                String startVertex = vertexList.get(vertexIdx);
                List<String> walk = generateRandomWalk(startVertex);
                allWalks.add(walk);
                walksGenerated++;

                // Print progress every 5 seconds or 10% increments
                long currentTime = System.currentTimeMillis();
                if (currentTime - lastPrintTime > 5000 ||
                        walksGenerated % (Math.max(1, totalWalks / 10)) == 0 ||
                        walksGenerated == totalWalks) {

                    double progress = (double) walksGenerated / totalWalks * 100;
                    System.out.printf("      📊 Progress: %d/%d walks (%.1f%%)%n",
                            walksGenerated, totalWalks, progress);
                    lastPrintTime = currentTime;
                }
            }
        }

        System.out.printf("✅ Random walk generation completed: %d walks generated%n", allWalks.size());
        return allWalks;
    }

    // Generate a single random walk starting from given vertex
    private List<String> generateRandomWalk(String startVertex) {
        List<String> walk = new ArrayList<>();
        walk.add(startVertex);

        String currentVertex = startVertex;
        for (int step = 1; step < walkLength; step++) {
            Set<String> neighbors = graph.getNeighbors(currentVertex);
            if (neighbors.isEmpty()) {
                break;
            }

            // Randomly select next vertex
            List<String> neighborList = new ArrayList<>(neighbors);
            String nextVertex = neighborList.get(random.nextInt(neighborList.size()));
            walk.add(nextVertex);
            currentVertex = nextVertex;
        }

        return walk;
    }

    // Initialize embeddings with random values
    private void initializeEmbeddings() {
        System.out.println("🎯 Initializing embeddings...");

        // Get all vertices from both authors and publications
        Set<String> allVertices = new HashSet<>(graph.getAuthorVertices());
        allVertices.addAll(graph.getPublicationVertices());

        int nodeCount = allVertices.size();
        int initialized = 0;

        for (String vertex : allVertices) {
            double[] embedding = new double[dimensions];
            for (int i = 0; i < dimensions; i++) {
                embedding[i] = (random.nextDouble() - 0.5) / dimensions;
            }
            embeddings.put(vertex, embedding);
            initialized++;

            // Print progress for large graphs
            if (nodeCount > 10000 && initialized % 10000 == 0) {
                System.out.printf("   📊 Initialized %d/%d nodes%n", initialized, nodeCount);
            }
        }
        System.out.printf("✅ Embeddings initialized for %d nodes%n", initialized);
    }

    // Train embeddings using Skip-gram with detailed progress
    public void train() {  // CHANGED FROM fit() TO train()
        System.out.println("\n🚀 Starting DeepWalk Training");
        System.out.println("================================");
        System.out.printf("Graph: %d nodes, %d edges%n", graph.getVertexCount(), graph.getEdgeCount());
        System.out.printf("Parameters: dimensions=%d, walkLength=%d, numWalks=%d, windowSize=%d, epochs=%d%n",
                dimensions, walkLength, numWalks, windowSize, epochs);

        startTime = System.currentTimeMillis();

        System.out.println("\n🎯 Phase 1: Initializing embeddings...");
        initializeEmbeddings();

        System.out.println("\n🎯 Phase 2: Generating random walks...");
        List<List<String>> walks = generateRandomWalks();

        System.out.println("\n🎯 Phase 3: Training skip-gram model...");
        trainSkipGram(walks);

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.printf("\n✅ DeepWalk training completed in %.2f seconds%n", totalTime / 1000.0);
    }

    // Enhanced training with epoch-level progress
    private void trainSkipGram(List<List<String>> walks) {
        System.out.printf("Training on %d walks with window size %d%n", walks.size(), windowSize);

        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.printf("\n📚 Epoch %d/%d%n", epoch + 1, epochs);
            long epochStartTime = System.currentTimeMillis();
            double totalLoss = 0;
            int processedWalks = 0;
            int totalSteps = 0;

            for (int walkIdx = 0; walkIdx < walks.size(); walkIdx++) {
                List<String> walk = walks.get(walkIdx);

                for (int centerPos = 0; centerPos < walk.size(); centerPos++) {
                    String centerVertex = walk.get(centerPos);

                    // Define context window
                    int start = Math.max(0, centerPos - windowSize);
                    int end = Math.min(walk.size() - 1, centerPos + windowSize);

                    for (int contextPos = start; contextPos <= end; contextPos++) {
                        if (contextPos == centerPos) continue;

                        String contextVertex = walk.get(contextPos);
                        double loss = updateEmbeddings(centerVertex, contextVertex);
                        totalLoss += loss;
                        totalSteps++;
                    }
                }
                processedWalks++;

                // Print progress every 1000 walks or 10% increments
                if (processedWalks % 1000 == 0 || processedWalks == walks.size()) {
                    double progress = (double) processedWalks / walks.size() * 100;
                    long currentTime = System.currentTimeMillis();
                    long elapsed = currentTime - epochStartTime;
                    double stepsPerSec = totalSteps / Math.max(1, (elapsed / 1000.0));

                    System.out.printf("   📊 Walk %d/%d (%.1f%%) - Steps/sec: %.0f - Avg Loss: %.6f%n",
                            processedWalks, walks.size(), progress, stepsPerSec,
                            totalSteps > 0 ? totalLoss / totalSteps : 0);
                }
            }

            long epochTime = System.currentTimeMillis() - epochStartTime;
            double avgLoss = totalSteps > 0 ? totalLoss / totalSteps : 0;
            System.out.printf("   ✅ Epoch completed in %.2f seconds - Final Loss: %.6f%n",
                    epochTime / 1000.0, avgLoss);
        }
    }

    // Update embeddings using gradient descent
    private double updateEmbeddings(String center, String context) {
        double[] centerEmbedding = embeddings.get(center);
        double[] contextEmbedding = embeddings.get(context);

        if (centerEmbedding == null || contextEmbedding == null) {
            return 0.0;
        }

        double dotProduct = 0.0;
        for (int i = 0; i < dimensions; i++) {
            dotProduct += centerEmbedding[i] * contextEmbedding[i];
        }

        // Sigmoid function
        double sigmoid = 1.0 / (1.0 + Math.exp(-Math.max(Math.min(dotProduct, 10), -10))); // Clipped for numerical stability

        // Gradient (simplified - using 1 as target since they co-occur)
        double gradient = (1 - sigmoid) * learningRate;

        // Update embeddings
        for (int i = 0; i < dimensions; i++) {
            double centerUpdate = gradient * contextEmbedding[i];
            double contextUpdate = gradient * centerEmbedding[i];

            centerEmbedding[i] += centerUpdate;
            contextEmbedding[i] += contextUpdate;
        }

        // Return cross-entropy loss
        return -Math.log(Math.max(sigmoid, 1e-10)); // Avoid log(0)
    }

    // Get cosine similarity between two vertices
    public double similarity(String vertex1, String vertex2) {
        double[] emb1 = embeddings.get(vertex1);
        double[] emb2 = embeddings.get(vertex2);

        if (emb1 == null || emb2 == null) {
            return -1.0;
        }

        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (int i = 0; i < dimensions; i++) {
            dotProduct += emb1[i] * emb2[i];
            norm1 += emb1[i] * emb1[i];
            norm2 += emb2[i] * emb2[i];
        }

        if (norm1 == 0 || norm2 == 0) {
            return 0.0;
        }

        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    // Find most similar vertices
    public List<Map.Entry<String, Double>> mostSimilar(String vertex, int topN) {
        System.out.printf("🔍 Finding %d most similar nodes to '%s'...%n", topN, vertex);

        List<Map.Entry<String, Double>> similarities = new ArrayList<>();
        Set<String> allNodes = new HashSet<>(embeddings.keySet());
        int processed = 0;

        for (String otherVertex : allNodes) {
            if (!otherVertex.equals(vertex)) {
                double sim = similarity(vertex, otherVertex);
                similarities.add(new AbstractMap.SimpleEntry<>(otherVertex, sim));
                processed++;

                if (processed % 10000 == 0) {
                    System.out.printf("   📊 Processed %d/%d comparisons%n", processed, allNodes.size() - 1);
                }
            }
        }

        similarities.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        System.out.printf("✅ Similarity search completed for %d nodes%n", processed);

        return similarities.subList(0, Math.min(topN, similarities.size()));
    }

    // Get embeddings
    public Map<String, double[]> getEmbeddings() {
        return new HashMap<>(embeddings);
    }

    // Save embeddings to file
    public void saveEmbeddings(String filename) {
        System.out.printf("💾 Saving embeddings to '%s'...%n", filename);
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            int saved = 0;
            for (Map.Entry<String, double[]> entry : embeddings.entrySet()) {
                writer.print(entry.getKey());
                for (double value : entry.getValue()) {
                    writer.printf(" %.6f", value);
                }
                writer.println();
                saved++;

                if (saved % 10000 == 0) {
                    System.out.printf("   📊 Saved %d/%d embeddings%n", saved, embeddings.size());
                }
            }
            System.out.printf("✅ Saved %d embeddings to '%s'%n", saved, filename);
        } catch (IOException e) {
            System.err.println("❌ Error saving embeddings: " + e.getMessage());
        }
    }
}