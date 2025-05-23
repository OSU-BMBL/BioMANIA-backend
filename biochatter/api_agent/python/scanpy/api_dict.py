import scanpy as sc

pp_api_list = [
	{
		"api": sc.pp.neighbors,
		"products": [
			"data.uns[\"neighbors\"]",
			"data.obsp[\"distances\"]",
			"data.obsp[\"connectivities\"]"
		],
		"data_name": "adata"
	},
	{
		"api": sc.pp.log1p,
		"products": [
			"data.X"
		],
		"data_name": "data"
	},
	{
		"api": sc.pp.highly_variable_genes,
		"products": [
			"data.var[\"highly_variable\"]",
			"data.var[\"means\"]",
			"data.var[\"dispersions\"]",
			"data.var[\"dispersions_norm\"]",
			"data.var[\"variances\"]",
			"data.var[\"variances_norm\"]",
			"data.var[\"highly_variable_rank\"]",
			"data.var[\"highly_variable_nbatches\"]",
			"data.var[\"highly_variable_intersection\"]"
		],
		"data_name": "adata",
		"_comment": "These products are curated from doc, from 'variances' to the end are actually non-existent.Products are not always fully generated, such as the current case.When testing dependencies, results would show non-existent dependencies. Although we already fixed this problem in dependency finder program,this is still a problematic software design making return values unclear and unfixed.We should encourage fixed api name, arguments and return values. In the meanwhile, returning an object is acceptable only if it is fixed. Note that we hope products can be automatically extracted from doc through LLM. If doc provides vague information products, then we believe it is poorly designed."
	},
	{
		"api": sc.pp.pca,
		"products": [
			"data.obsm[\"X_pca\"]",
			"data.varm[\"PCs\"]",
			"data.uns[\"pca\"][\"variance_ratio\"]",
			"data.uns[\"pca\"][\"variance\"]"
		],
		"data_name": "data"
	},
]

tl_api_list = [
	{
		"api": sc.tl.paga,
		"products": [
			"data.uns[\"paga\"][\"connectivities\"]",
			"data.uns[\"paga\"][\"connectivities_tree\"]"
		],
		"data_name": "adata"
	},
	{
		"api": sc.tl.leiden,
		"products": [
			"data.obs[\"leiden\"]",
			"data.uns[\"leiden\"]"
		],
		"data_name": "adata"
	},
	{
		"api": sc.tl.louvain,
		"products": [
			"data.obs[\"louvain\"]",
			"data.uns[\"louvain\"]"
		],
		"data_name": "adata"
	},
	{
		"api": sc.tl.umap,
		"products": [
			"data.obsm[\"X_umap\"]",
			"data.uns[\"umap\"]"
		],
		"data_name": "adata"
	},
	{
		"api": sc.tl.tsne,
		"products": [
			"data.obsm[\"X_tsne\"]",
			"data.uns[\"tsne\"]"
		],  
		"data_name": "adata"
	},
	{
		"api": sc.tl.diffmap,
		"products": [
			"data.obsm[\"X_diffmap\"]",
			"data.uns[\"diffmap_evals\"]"
		],
		"data_name": "adata"
	},
	{
		"api": sc.tl.embedding_density,
		"products": [
			"data.obs[\"umap_density\"]",
			"data.uns[\"umap_density_params\"]"
		],
		"data_name": "adata"
	},
	{
		"api": sc.tl.rank_genes_groups,
		"products": [
			"data.uns[\"rank_genes_groups\"]"
		],
		"data_name": "adata"
	},
]

pl_api_list = [
	{
		"api": sc.pl.paga,
		"products": [
			"data.uns[\"paga\"][\"pos\"]"
		],
		"data_name": "adata"
	},
	{
		"api": sc.pl.scatter,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.umap,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.tsne,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.heatmap,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.dotplot,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.violin,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.dendrogram,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.diffmap,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.highly_variable_genes,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.pca,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.embedding_density,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.rank_genes_groups_dotplot,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.correlation_matrix,
		"products": [],
		"data_name": "adata",
		"_deprecated": True,
		"_comment": "Not found in doc."
	},
	{
		"api": sc.pl.highest_expr_genes,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.tracksplot,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.clustermap,
		"products": [],
		"data_name": "adata"
	},
	{
		"api": sc.pl.stacked_violin,
		"products": [],
		"data_name": "adata"
	}
]

pp = {
	"meta": {
		"package": sc,
		"module": sc.pp,
	},
	"api_list": pp_api_list
}

tl = {
	"meta": {
		"package": sc,
		"module": sc.tl,
	},
	"api_list": tl_api_list
}

pl = {
	"meta": {
		"package": sc,
		"module": sc.pl,
	},
	"api_list": pl_api_list
}