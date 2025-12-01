package org.example.Fetch;

public class Vertex {
    private String id;
    private String type; // "author" or "publication"
    private Object data; // Reference to Author or Publication object

    public Vertex(String id, String type, Object data) {
        this.id = id;
        this.type = type;
        this.data = data;
    }

    public String getId() { return id; }
    public String getType() { return type; }
    public Object getData() { return data; }

    @Override
    public String toString() {
        return "Vertex{" + "id='" + id + "', type='" + type + "'}";
    }
}