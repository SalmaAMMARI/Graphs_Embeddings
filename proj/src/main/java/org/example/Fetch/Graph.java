package org.example.Fetch;

import java.util.*;

public class Graph {
    private final Map<String, Vertex> vertices; // All vertices
    private final Set<String> authorVertices; // Only author vertices
    private final Set<String> publicationVertices; // Only publication vertices
    private final Map<String, Set<String>> edges; // Adjacency list for edges - CHANGED FROM adjacencyList TO edges

    public Graph() {
        this.vertices = new HashMap<>();
        this.authorVertices = new HashSet<>();
        this.publicationVertices = new HashSet<>();
        this.edges = new HashMap<>(); // CHANGED TO edges
    }

    // Add author vertex
    public void addAuthorVertex(String name, Author author) {
        Vertex vertex = new Vertex(name, "author", author);
        vertices.put(name, vertex);
        authorVertices.add(name);
        if (!edges.containsKey(name)) { // CHANGED TO edges
            edges.put(name, new HashSet<>());
        }
    }

    // Add publication vertex
    public void addPublicationVertex(String key, Publication publication) {
        Vertex vertex = new Vertex(key, "publication", publication);
        vertices.put(key, vertex);
        publicationVertices.add(key);
        if (!edges.containsKey(key)) { // CHANGED TO edges
            edges.put(key, new HashSet<>());
        }
    }

    // Add edge between author and publication
    public void addEdge(String authorName, String publicationKey) {
        if (edges.containsKey(authorName) && edges.containsKey(publicationKey)) { // CHANGED TO edges
            edges.get(authorName).add(publicationKey);
            edges.get(publicationKey).add(authorName);
        }
    }

    // Getters
    public int getVertexCount() {
        return vertices.size();
    }

    public int getEdgeCount() {
        int count = 0;
        for (Set<String> neighbors : edges.values()) { // CHANGED TO edges
            count += neighbors.size();
        }
        return count / 2; // Each edge is counted twice (undirected)
    }

    public Set<String> getAuthorVertices() {
        return new HashSet<>(authorVertices);
    }

    public Set<String> getPublicationVertices() {
        return new HashSet<>(publicationVertices);
    }

    public Map<String, Set<String>> getAdjacencyList() {
        return new HashMap<>(edges); // CHANGED TO edges
    }

    public Set<String> getNeighbors(String vertex) {
        return edges.getOrDefault(vertex, new HashSet<>()); // CHANGED TO edges
    }

    // Find maximum matching using a greedy approach
    public Map<String, String> findMaximumMatching() {
        Map<String, String> matching = new HashMap<>();
        Set<String> matchedVertices = new HashSet<>();

        // Try to match each author with an unmatched publication
        for (String author : authorVertices) {
            if (matchedVertices.contains(author)) continue;

            for (String publication : getNeighbors(author)) {
                if (!matchedVertices.contains(publication)) {
                    matching.put(author, publication);
                    matchedVertices.add(author);
                    matchedVertices.add(publication);
                    break; // Match this author with the first available publication
                }
            }
        }

        return matching;
    }

    // Find maximum independent set (greedy approach)
    public Set<String> findMaximumIndependentSet(String vertexType) {
        Set<String> verticesToConsider = "author".equals(vertexType) ?
                new HashSet<>(authorVertices) : new HashSet<>(publicationVertices);

        Set<String> independentSet = new HashSet<>();
        Set<String> coveredVertices = new HashSet<>();

        // For bipartite graphs, the maximum independent set can be found using
        // maximum matching and König's theorem
        // But for a simple greedy approach:
        for (String vertex : verticesToConsider) {
            if (!coveredVertices.contains(vertex)) {
                independentSet.add(vertex);
                // Add this vertex and its neighbors to covered set
                coveredVertices.add(vertex);
                coveredVertices.addAll(getNeighbors(vertex));
            }
        }

        return independentSet;
    }

    // Check if the graph is bipartite using BFS
    public boolean isBipartite() {
        Map<String, Integer> color = new HashMap<>();

        // Initialize all vertices with -1 (unvisited)
        for (String vertex : vertices.keySet()) {
            color.put(vertex, -1);
        }

        // Check each component of the graph
        for (String startVertex : vertices.keySet()) {
            if (color.get(startVertex) != -1) continue; // Already visited

            Queue<String> queue = new LinkedList<>();
            queue.add(startVertex);
            color.put(startVertex, 0); // Color with 0

            while (!queue.isEmpty()) {
                String current = queue.poll();

                for (String neighbor : getNeighbors(current)) {
                    if (color.get(neighbor) == -1) {
                        // Color with opposite color
                        color.put(neighbor, 1 - color.get(current));
                        queue.add(neighbor);
                    } else if (color.get(neighbor) == color.get(current)) {
                        // Same color as current vertex - not bipartite
                        return false;
                    }
                }
            }
        }

        return true;
    }

    // Get degree of a vertex
    public int getDegree(String vertex) {
        return getNeighbors(vertex).size();
    }

    // Find vertices with highest degree
    public List<String> findHighDegreeVertices(int topN) {
        List<Map.Entry<String, Integer>> degreeList = new ArrayList<>();
        for (String vertex : vertices.keySet()) {
            degreeList.add(new AbstractMap.SimpleEntry<>(vertex, getDegree(vertex)));
        }

        degreeList.sort((a, b) -> Integer.compare(b.getValue(), a.getValue()));

        List<String> result = new ArrayList<>();
        for (int i = 0; i < Math.min(topN, degreeList.size()); i++) {
            result.add(degreeList.get(i).getKey());
        }

        return result;
    }
}