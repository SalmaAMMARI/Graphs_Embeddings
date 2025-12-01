package org.example.Fetch;

import java.util.Objects;
import java.util.Set;
import java.util.HashSet;

public class Author {
    private String name;
    private Set<String> publicationKeys;

    public Author(String name) {
        this.name = name;
        this.publicationKeys = new HashSet<>();
    }

    public String getName() { return name; }
    public Set<String> getPublicationKeys() { return new HashSet<>(publicationKeys); }

    public void addPublicationKey(String key) {
        if (key != null && !key.trim().isEmpty()) {
            publicationKeys.add(key);
        }
    }

    @Override
    public String toString() {
        return "Author{" +
                "name='" + name + '\'' +
                ", publications=" + publicationKeys.size() + " keys" +
                '}';
    }
}