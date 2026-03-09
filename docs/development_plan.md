# SlideCraft AI - Development Plan

## Project Vision

An agentic application that generates professional presentations from source documents using RAG-based retrieval and AI-powered content generation.

---

### Phase 1: Data Ingestion Pipeline

**Components**:
- **PDFLoader** (`FileLoader`): Extracts text content from PDF documents using PyMuPDF
- **TextChunker**: Splits document text into semantic chunks with overlap tracking
- **PDFImageExtractor**: Extracts images from PDF pages
- **TextEmbedder**: Generates 384-dim embeddings using sentence-transformers
- **ImageEmbedder**: Generates 512-dim embeddings using CLIP ViT-B/32
- **Storage**: File and image storage with volume persistence

### Phase 2: RAG Retrieval + Context Assembly

**Components**:
- **SearchService**: Performs semantic search over both text chunks (384-dim) and images (512-dim)
  - Text-to-text search using sentence-transformers
  - Text-to-image cross-modal search using CLIP
  - Configurable limits and similarity thresholds
  
- **ContextAssembler**: Transforms raw search results into structured, document-ordered context
  - Groups chunks by page number
  - Merges overlapping/adjacent chunks into coherent passages
  - Attaches semantically relevant images to passages
  - Maintains document order (page-by-page)

---

### Phase 3: Presentation Structure Planning (Agent Layer 1)

**Goal**: Intelligent analysis of retrieval context to plan presentation structure

**Components to Build**:

1. **ContentAnalyzer**
   - Analyzes `RetrievalContext` to identify key themes, topics, and concepts
   - Extracts main ideas, supporting details, and relationships
   - Identifies which passages/images are most relevant for presentation
   - Categorizes content by type (introduction, data, conclusions, etc.)

2. **PresentationPlanner** (Agentic)
   - Takes analyzed content and generates presentation outline
   - Determines optimal number of slides
   - Plans slide types (title, content, image, data visualization, etc.)
   - Creates logical flow and narrative structure
   - Assigns passages and images to specific slides
   - Uses LLM reasoning to make strategic decisions about content organization

---

### Phase 4: Slide Content Generation (Agent Layer 2)

**Goal**: Generate actual slide content (text, bullet points, titles) for each planned slide

**Components to Build**:

1. **SlideContentGenerator** (Agentic)
   - Takes `SlideOutline` and source passages
   - Generates concise, presentation-ready text
   - Creates compelling slide titles
   - Formats bullet points and key messages
   - Adapts tone and style based on preferences
   - Ensures consistency across slides

2. **ImageSelector**
   - Selects best images for each slide from available matches
   - Considers semantic relevance, page context, and visual balance
   - Handles image placement recommendations

3. **ContentRefiner**
   - Ensures generated content fits slide constraints
   - Validates consistency and flow
   - Checks for redundancy across slides

---

### Phase 5: Presentation Rendering & Export

**Goal**: Convert generated content into actual presentation files

**Components to Build**:

1. **PresentationRenderer**
   - Takes `GeneratedPresentation` and renders to file format
   - Applies templates and themes
   - Handles layout and positioning
   - Embeds images with proper sizing

2. **TemplateManager**
   - Manages presentation templates (corporate, minimal, creative, etc.)
   - Handles theme customization (colors, fonts, layouts)
   - Provides template selection based on content type


---

### Phase 6: Agentic Orchestration & Refinement

**Goal**: End-to-end agentic workflow with iterative refinement

**Components to Build**:

1. **PresentationAgent** (Master Orchestrator)
   - Coordinates all phases from query to final presentation
   - Makes high-level decisions about presentation strategy
   - Handles error recovery and retries
   - Manages conversation state for iterative refinement

2. **FeedbackProcessor**
   - Accepts user feedback on generated presentations
   - Identifies specific slides or content to modify
   - Generates refinement instructions for content generator

3. **IterativeRefiner** (Agentic)
   - Takes feedback and regenerates specific slides
   - Maintains consistency with unchanged slides
   - Learns from user preferences over time

4. **QualityValidator**
   - Validates generated presentations for quality
   - Checks for common issues (too much text, poor flow, missing context)
   - Suggests improvements before final render
