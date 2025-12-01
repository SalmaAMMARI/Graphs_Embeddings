package org.example.Fetch;

import java.util.*;
import java.util.stream.Collectors;

public class LinkPredictor {
    private final Map<String, double[]> embeddings;
    private final Graph graph;
    private final Random random;

    public LinkPredictor(Map<String, double[]> embeddings, Graph graph) {
        this.embeddings = embeddings;
        this.graph = graph;
        this.random = new Random(42);
    }

    public void evaluateLinkPrediction(int topK) {
        System.out.println("\n🔗 Starting Link Prediction Evaluation");
        System.out.println("=====================================");
        System.out.printf("Evaluating top-%d predictions%n", topK);
        System.out.printf("Graph: %d nodes, %d edges%n", graph.getVertexCount(), graph.getEdgeCount());
        System.out.printf("Embeddings: %d nodes with vectors%n", embeddings.size());

        long startTime = System.currentTimeMillis();

        // Split edges into training and test sets
        System.out.println("\n🎯 Phase 1: Splitting edges into train/test sets...");
        Map<String, Set<String>> trainGraph = new HashMap<>();
        Set<String> testEdges = new HashSet<>();
        List<String> allEdges = getAllEdges();

        splitEdges(trainGraph, testEdges, allEdges, 0.8); // 80% train, 20% test

        System.out.printf("   Training edges: %d%n", countEdges(trainGraph));
        System.out.printf("   Test edges: %d%n", testEdges.size());

        // Generate negative samples
        System.out.println("\n🎯 Phase 2: Generating negative samples...");
        Set<String> negativeEdges = generateNegativeSamples(testEdges.size() * 2); // 2:1 negative:positive ratio
        System.out.printf("   Generated %d negative edges%n", negativeEdges.size());

        // Evaluate predictions
        System.out.println("\n🎯 Phase 3: Evaluating link predictions...");
        evaluatePredictions(trainGraph, testEdges, negativeEdges, topK);

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.printf("\n✅ Link prediction evaluation completed in %.2f seconds%n", totalTime / 1000.0);
    }

    private List<String> getAllEdges() {
        System.out.println("   Collecting all edges from graph...");
        List<String> allEdges = new ArrayList<>();
        int collected = 0;

        for (String author : graph.getAuthorVertices()) {
            for (String publication : graph.getNeighbors(author)) {
                String edge = author + "|" + publication;
                allEdges.add(edge);
                collected++;

                if (collected % 100000 == 0) {
                    System.out.printf("      📊 Collected %d edges%n", collected);
                }
            }
        }

        System.out.printf("   ✅ Total edges collected: %d%n", collected);
        return allEdges;
    }

    private void splitEdges(Map<String, Set<String>> trainGraph, Set<String> testEdges,
                            List<String> allEdges, double trainRatio) {
        System.out.println("   Shuffling and splitting edges...");
        Collections.shuffle(allEdges, random);

        int trainSize = (int) (allEdges.size() * trainRatio);
        int processed = 0;

        for (int i = 0; i < allEdges.size(); i++) {
            String edge = allEdges.get(i);
            String[] parts = edge.split("\\|");
            String node1 = parts[0];
            String node2 = parts[1];

            if (i < trainSize) {
                // Add to training graph
                trainGraph.computeIfAbsent(node1, k -> new HashSet<>()).add(node2);
                trainGraph.computeIfAbsent(node2, k -> new HashSet<>()).add(node1);
            } else {
                // Add to test set
                testEdges.add(edge);
            }

            processed++;
            if (processed % 100000 == 0) {
                System.out.printf("      📊 Processed %d/%d edges (%.1f%%)%n",
                        processed, allEdges.size(), (double)processed/allEdges.size()*100);
            }
        }

        System.out.printf("   ✅ Edge splitting completed: %d train, %d test%n",
                countEdges(trainGraph), testEdges.size());
    }

    private int countEdges(Map<String, Set<String>> graph) {
        int count = 0;
        for (Set<String> neighbors : graph.values()) {
            count += neighbors.size();
        }
        return count / 2; // Undirected graph
    }

    private Set<String> generateNegativeSamples(int numSamples) {
        System.out.printf("   Generating %d negative samples...%n", numSamples);
        Set<String> negativeEdges = new HashSet<>();
        List<String> allNodes = new ArrayList<>(embeddings.keySet());
        int generated = 0;
        int attempts = 0;
        int maxAttempts = numSamples * 10; // Prevent infinite loops

        while (generated < numSamples && attempts < maxAttempts) {
            String node1 = allNodes.get(random.nextInt(allNodes.size()));
            String node2 = allNodes.get(random.nextInt(allNodes.size()));

            // Ensure it's not an existing edge and nodes are different
            if (!node1.equals(node2) && !graph.getNeighbors(node1).contains(node2)) {
                String edge = node1 + "|" + node2;
                if (negativeEdges.add(edge)) {
                    generated++;

                    if (generated % 1000 == 0) {
                        System.out.printf("      📊 Generated %d/%d negative samples%n", generated, numSamples);
                    }
                }
            }
            attempts++;
        }

        System.out.printf("   ✅ Generated %d negative samples (%d attempts)%n", generated, attempts);
        return negativeEdges;
    }

    private void evaluatePredictions(Map<String, Set<String>> trainGraph,
                                     Set<String> testEdges, Set<String> negativeEdges, int topK) {
        System.out.println("   Calculating similarity scores for all node pairs...");

        // Combine test and negative edges
        Set<String> allEvaluationEdges = new HashSet<>();
        allEvaluationEdges.addAll(testEdges);
        allEvaluationEdges.addAll(negativeEdges);

        // Calculate scores
        List<EdgeScore> allScores = new ArrayList<>();
        int processed = 0;
        int totalEdges = allEvaluationEdges.size();

        for (String edge : allEvaluationEdges) {
            String[] parts = edge.split("\\|");
            String node1 = parts[0];
            String node2 = parts[1];

            double score = cosineSimilarity(embeddings.get(node1), embeddings.get(node2));
            boolean isPositive = testEdges.contains(edge);

            allScores.add(new EdgeScore(node1, node2, score, isPositive));
            processed++;

            if (processed % 1000 == 0 || processed == totalEdges) {
                System.out.printf("      📊 Scored %d/%d edges (%.1f%%)%n",
                        processed, totalEdges, (double)processed/totalEdges*100);
            }
        }

        // Sort by score descending
        System.out.println("   Sorting edges by similarity score...");
        allScores.sort((a, b) -> Double.compare(b.score, a.score));

        // Calculate metrics
        System.out.println("   Calculating evaluation metrics...");
        calculateMetrics(allScores, testEdges.size(), topK);
    }

    private void calculateMetrics(List<EdgeScore> allScores, int numPositives, int topK) {
        int truePositives = 0;
        int falsePositives = 0;
        int totalPositives = numPositives;
        int totalNegatives = allScores.size() - numPositives;

        double areaUnderCurve = 0;
        int previousTruePositives = 0;
        int previousFalsePositives = 0;

        List<Double> precisions = new ArrayList<>();
        List<Double> recalls = new ArrayList<>();

        System.out.println("\n   📈 Calculating ROC and Precision-Recall curves...");

        for (int i = 0; i < allScores.size(); i++) {
            EdgeScore score = allScores.get(i);

            if (score.isPositive) {
                truePositives++;
            } else {
                falsePositives++;
            }

            // Calculate AUC using trapezoidal rule
            if (i > 0) {
                double falsePositiveRate = (double) falsePositives / totalNegatives;
                double previousFalsePositiveRate = (double) previousFalsePositives / totalNegatives;
                double truePositiveRate = (double) truePositives / totalPositives;
                double previousTruePositiveRate = (double) previousTruePositives / totalPositives;

                areaUnderCurve += 0.5 * (falsePositiveRate - previousFalsePositiveRate) *
                        (truePositiveRate + previousTruePositiveRate);
            }

            // Store precision and recall for top-K
            if (i < topK) {
                double precision = (double) truePositives / (i + 1);
                double recall = (double) truePositives / totalPositives;
                precisions.add(precision);
                recalls.add(recall);
            }

            previousTruePositives = truePositives;
            previousFalsePositives = falsePositives;

            // Progress for large datasets
            if (allScores.size() > 10000 && (i + 1) % (allScores.size() / 10) == 0) {
                System.out.printf("      📊 Processed %d/%d scores (%.1f%%)%n",
                        i + 1, allScores.size(), (double)(i + 1)/allScores.size()*100);
            }
        }

        // Final metrics
        double precisionAtK = precisions.isEmpty() ? 0 : precisions.get(precisions.size() - 1);
        double recallAtK = recalls.isEmpty() ? 0 : recalls.get(recalls.size() - 1);
        double f1AtK = (precisionAtK + recallAtK == 0) ? 0 :
                2 * (precisionAtK * recallAtK) / (precisionAtK + recallAtK);

        // Print results
        System.out.println("\n📊 Link Prediction Results:");
        System.out.println("==========================");
        System.out.printf("AUC (Area Under ROC Curve): %.4f%n", areaUnderCurve);
        System.out.printf("Precision@%d: %.4f%n", topK, precisionAtK);
        System.out.printf("Recall@%d: %.4f%n", topK, recallAtK);
        System.out.printf("F1-Score@%d: %.4f%n", topK, f1AtK);
        System.out.printf("True Positives in top-%d: %d/%d%n", topK, truePositives, totalPositives);

        // Print precision-recall curve points
        System.out.println("\nPrecision-Recall Curve (top-" + topK + "):");
        for (int i = 0; i < Math.min(5, precisions.size()); i++) {
            System.out.printf("  Position %d: Precision=%.3f, Recall=%.3f%n",
                    i + 1, precisions.get(i), recalls.get(i));
        }
        if (precisions.size() > 5) {
            System.out.printf("  ... (showing first 5 of %d positions)%n", precisions.size());
        }
    }

    public void printTopPredictions(String node, int topN) {
        System.out.printf("\n🔍 Generating top-%d link predictions for: %s%n", topN, node);
        System.out.println("===========================================");

        if (!embeddings.containsKey(node)) {
            System.out.println("❌ Node not found in embeddings: " + node);
            return;
        }

        System.out.println("   Calculating similarities to all other nodes...");
        List<Map.Entry<String, Double>> predictions = new ArrayList<>();
        List<String> allNodes = new ArrayList<>(embeddings.keySet());
        int processed = 0;

        for (String otherNode : allNodes) {
            if (!otherNode.equals(node) && !graph.getNeighbors(node).contains(otherNode)) {
                double similarity = cosineSimilarity(embeddings.get(node), embeddings.get(otherNode));
                predictions.add(new AbstractMap.SimpleEntry<>(otherNode, similarity));
            }
            processed++;

            if (processed % 10000 == 0) {
                System.out.printf("      📊 Processed %d/%d nodes (%.1f%%)%n",
                        processed, allNodes.size(), (double)processed/allNodes.size()*100);
            }
        }

        // Sort by similarity descending
        System.out.println("   Sorting predictions by similarity...");
        predictions.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));

        // Print top predictions
        System.out.printf("\n🏆 Top-%d Link Predictions for %s:%n", topN, node);
        System.out.println("─────────────────────────────────────");

        int count = 0;
        for (Map.Entry<String, Double> prediction : predictions) {
            if (count >= topN) break;

            String otherNode = prediction.getKey();
            double similarity = prediction.getValue();
            String nodeType = graph.getAuthorVertices().contains(otherNode) ? "Author" : "Publication";
            String existingConnection = graph.getNeighbors(node).contains(otherNode) ? " ✅" : " 🔮";

            System.out.printf("%2d. %s (%.4f) [%s]%s%n",
                    count + 1, otherNode, similarity, nodeType, existingConnection);
            count++;
        }

        System.out.printf("✅ Generated %d predictions for node %s%n", Math.min(topN, predictions.size()), node);
    }

    private double cosineSimilarity(double[] vec1, double[] vec2) {
        if (vec1 == null || vec2 == null || vec1.length != vec2.length) {
            return 0.0;
        }

        double dotProduct = 0.0;
        double norm1 = 0.0;
        double norm2 = 0.0;

        for (int i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }

        if (norm1 == 0 || norm2 == 0) {
            return 0.0;
        }

        return dotProduct / (Math.sqrt(norm1) * Math.sqrt(norm2));
    }

    // Helper class for edge scores
    private static class EdgeScore {
        String node1;
        String node2;
        double score;
        boolean isPositive;

        EdgeScore(String node1, String node2, double score, boolean isPositive) {
            this.node1 = node1;
            this.node2 = node2;
            this.score = score;
            this.isPositive = isPositive;
        }
    }
}