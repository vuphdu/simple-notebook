# Sequence Chart Examples

This document contains example sequence diagrams for testing the chart processing module.

## User Authentication Flow

The following sequence diagram shows a typical user authentication flow:

```mermaid
sequenceDiagram
    title User Authentication Flow
    participant User
    participant Client
    participant AuthServer
    participant Database
    
    User->>Client: Enter credentials
    Client->>AuthServer: POST /login
    AuthServer->>Database: Validate credentials
    Database-->>AuthServer: User data
    AuthServer-->>Client: JWT Token
    Client-->>User: Login successful
```

## API Request Flow

Here's an example of a typical API request flow:

```mermaid
sequenceDiagram
    title API Request Processing
    participant Frontend
    participant APIGateway
    participant Backend
    participant Cache
    participant DB
    
    Frontend->>APIGateway: HTTP Request
    APIGateway->>Backend: Forward request
    Backend->>Cache: Check cache
    Cache-->>Backend: Cache miss
    Backend->>DB: Query data
    DB-->>Backend: Return data
    Backend->>Cache: Update cache
    Backend-->>APIGateway: Response
    APIGateway-->>Frontend: HTTP Response
```

## RAG System Flow

This diagram shows how the RAG system processes queries:

```mermaid
sequenceDiagram
    title RAG Query Processing
    participant User
    participant SearchEngine
    participant Vectorizer
    participant VectorDB
    participant ResultFormatter
    
    User->>SearchEngine: Submit query
    SearchEngine->>Vectorizer: Convert to embedding
    Vectorizer-->>SearchEngine: Query vector
    SearchEngine->>VectorDB: Similarity search
    VectorDB-->>SearchEngine: Top K results
    SearchEngine->>ResultFormatter: Format results
    ResultFormatter-->>User: Display results
```

## Description

These sequence diagrams illustrate common patterns in software systems:

1. **Authentication**: Shows the typical login flow with credential validation
2. **API Request**: Demonstrates caching strategy for API responses
3. **RAG Pipeline**: Illustrates the query processing in a RAG system

Each diagram can be exported as an image, and its description will be vectorized for semantic search.
