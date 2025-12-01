package org.example.Fetch;


import java.util.ArrayList;
import java.util.List;

public class Publication {
    private String key;
    private String title;
    private String year;
    private String pages;
    private String booktitle;
    private String ee;
    private String url;
    private String mdate;
    private String publtype;
    private String type; // incollection, book, phdthesis
    private List<String> authorNames; // Store author names

    public Publication(String key, String title, String year, String pages,
                       String booktitle, String ee, String url, String mdate,
                       String publtype, String type) {
        this.key = key;
        this.title = title;
        this.year = year;
        this.pages = pages;
        this.booktitle = booktitle;
        this.ee = ee;
        this.url = url;
        this.mdate = mdate;
        this.publtype = publtype;
        this.type = type;
        this.authorNames = new ArrayList<>(); // Initialize the list
    }

    // Add this method to add authors to the publication
    public void addAuthor(String authorName) {
        if (authorName != null && !authorName.trim().isEmpty()) {
            this.authorNames.add(authorName);
        }
    }

    // Getters and setters
    public String getKey() { return key; }
    public void setKey(String key) { this.key = key; }

    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }

    public String getYear() { return year; }
    public void setYear(String year) { this.year = year; }

    public String getPages() { return pages; }
    public void setPages(String pages) { this.pages = pages; }

    public String getBooktitle() { return booktitle; }
    public void setBooktitle(String booktitle) { this.booktitle = booktitle; }

    public String getEe() { return ee; }
    public void setEe(String ee) { this.ee = ee; }

    public String getUrl() { return url; }
    public void setUrl(String url) { this.url = url; }

    public String getMdate() { return mdate; }
    public void setMdate(String mdate) { this.mdate = mdate; }

    public String getPubltype() { return publtype; }
    public void setPubltype(String publtype) { this.publtype = publtype; }

    public String getType() { return type; }
    public void setType(String type) { this.type = type; }

    public List<String> getAuthorNames() { return new ArrayList<>(authorNames); }

    @Override
    public String toString() {
        return "Publication{" +
                "key='" + key + '\'' +
                ", title='" + title + '\'' +
                ", year='" + year + '\'' +
                ", authors=" + authorNames.size() +
                ", type='" + type + '\'' +
                '}';
    }
}