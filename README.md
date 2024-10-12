This is a description of the implementation of research accepted for publication at the IEEE ICECCME 2024.

Reference paper: D. Sweidan (2024). "Accurate, Even at Random: Exploring Multiple Distinct Data Clusterings via Random Projection". 
In proceeding: The IEEE 4th International Conference on Electrical, Computer, Communications and Mechatronics Engineering (ICECCME), November 04-06, 2024, Maldives.

The project is about a method for exploring different hidden insights in one given data, i.e., discovering different clustering solutions in the same data set. The approach employs random projection, where the original data is transformed (projected) multiple times into other spaces with randomly defined and linearly independent basis vectors. Each projection leads to obtaining one clustering solution using, e.g., a Gaussian Mixture Model with a specified number of components. The obtained clustering solutions from the multiple projections are further clustered using a hierarchical agglomerative clustering approach to group similar solutions. Each group of these solutions is aggregated into one representative clustering of the original data. Ultimately, the method explores several distinct and independent representative solutions that can be presented to the user.

To support the research community, the repository will include detailed documentation and examples to help users understand and apply the method. 
We are working on the optimization phase for large-dimensional data processing efficiency. We encourage others to use and extend our work.

The author.
