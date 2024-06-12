### Drawbacks of RecursiveCharacterTextSplitter
* Context loss: supposs context got splitted into multiple chunks then it will lead to less accurate output.
* Overhead of Recursive Splitting: Recursive splitting can be computationally intensive.
* Complexity in Implementation: developers have to find out the optimal chunk size and overlapping size number.
* Difficulty in Handling Edge Cases: In unstructured file, like having tables, images, list, bullet points,f&q, might lead to bad text splitting 

### Documents Well-Suited for RecursiveCharacterTextSplitter
* That have clear hierarchical structures or natural break points.
* Academic Papers and Articles
* Novel & Books
* Technical Manuals and Guides
* Web Articles and Blog Posts

