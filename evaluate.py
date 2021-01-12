import numpy as np
import pandas as pd
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from goatools.base import download_ncbi_associations
from goatools.anno.genetogo_reader import Gene2GoReader
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import utils


import os


gene2go = download_ncbi_associations()

objanno = Gene2GoReader(gene2go, taxids=[9606], go2geneids=True)

go2geneIDs = objanno.get_goid2dbids(objanno.associations) # this is a dict. Keys are GO IDs, values are gene_IDs of the genes that are associated to that GO term

geneID2GO = objanno.get_dbid2goids(objanno.associations)

goID2goTerm = {item.GO_ID :item.GO_term for item in objanno.associations}

genes_in_GO = list(geneID2GO.keys())  # these are entrez_ids




def distance_df(emb_df, metric='euclidean'):
    """Creates a distance matrix for a given embedding DataFrame.

    Args:
        emb_df (DataFrame): A DataFrame of shape (n_probes, n_features)
        metric (str, optional): Distance metric, defaults to 'euclidean'. Can also compute cosine similarity.

    Returns:
        dist (DataFrame): A square DataFrame of shape (n_probes, n_probes)
    """
    if metric == 'euclidean':
        #dist = euclidean_distances(emb_df)
        #dist = euclidean_distances(emb_df, emb_df)

        # for ISH embeddings, needs to be done this way to avoid calculating distance between the IDs
        dist = euclidean_distances(emb_df.iloc[:, 1:], emb_df.iloc[:, 1:])

    elif metric == 'cosine':
        dist = cosine_similarity(emb_df)

    dist = pd.DataFrame(dist)
    dist.index = emb_df.index
    dist.columns = emb_df.index

    return dist


def get_proportion_first_match(emb_df, metric='euclidean'):
    """Operates on probe embedding and checks to see if nearest probe judged by distance metric is of another probe for the same gene.

    Args:
        emb_df (pd.DataFrame): A DataFrame of shape (n_samples, n_features)
        metric (str, optional): Distance metric, defaults to 'euclidean'. Can also compute cosine similarity.

    Returns:
        float: proportion of probes that match to another probe of the same gene
    """
    dist = distance_df(emb_df, metric=metric)

    if metric == 'euclidean':
        np.fill_diagonal(dist.values, float('inf'))
        closest_indexes = dist.idxmin(axis=1, skipna=True).reset_index()
    elif metric == 'cosine':
        np.fill_diagonal(dist.values, float('0'))
        closest_indexes = dist.idxmax(axis=1, skipna=True).reset_index()

    closest_indexes.columns = ['probe_id', 'neighbor']
    closest_indexes['gene_id'] = closest_indexes.probe_id.map(probe_map)
    closest_indexes['nearest_gene'] = closest_indexes.neighbor.map(probe_map)
    proportion_closest_match = closest_indexes[closest_indexes.gene_id == closest_indexes.nearest_gene].shape[0] / closest_indexes.shape[0]

    return proportion_closest_match


def get_closest_probes(emb_df, probe_id, metric='euclidean'):
    dist = distance_df(emb_df, metric=metric)
    dist.index.name = 'probe_id'

    result = dist[dist.index == probe_id].iloc[0, :].sort_values()
    result = result.reset_index()
    result['gene'] = result.probe_id.map(probe_map)

    return result


def calc_probe_match_auc(emb_df, mask, probe_map='default', metric='euclidean'):
    """Calculates AUC where matches are for different probes of same gene symbol.

    Args:
        emb_df (pd.DataFrame): A DataFrame of shape (n_samples, n_features)
        probe_map (str, optional): Mapping of probes to gene symbols. Default is from Allen fetal brain.
        metric (str, optional): Defaults to 'euclidean'.

    Returns:
        auc (float)
    """

    if probe_map == 'default':
        probe_ids = pd.read_csv('./data/raw/allen_human_fetal_brain/lmd_matrix_12566/rows_metadata.csv', usecols=['probeset_name', 'gene_symbol'])
        probe_ids = probe_ids.set_index('probeset_name').to_dict()['gene_symbol']

    elif probe_map == 'reannotator':
        # the following is to map probes in the same manner as was done while training NN/embeddings
        probe_ids = pd.read_table('./data/raw/gene_symbol_annotations/AllenInstitute_custom_Agilent_Array.txt', sep='\t')
        probe_ids = probe_ids.rename(columns={'#PROBE_ID': 'probe_id',
                                              'Gene_symbol': 'gene_symbol'}).loc[:, ['probe_id', 'gene_symbol']]
        probe_ids.gene_symbol = probe_ids.gene_symbol.str.split(';').str[0]
        probe_ids = probe_ids.set_index('probe_id').to_dict()['gene_symbol']
    else:
        raise ValueError("Error: specify probe_map as either 'default' or 'reannotator'.")

    dist = distance_df(emb_df)
    dist.index.name = 'probe_id'
    np.fill_diagonal(dist.values, float('inf'))
    #dist.drop('#na#', inplace=True)
    #dist.drop('#na#', axis=1, inplace=True)
    dist = dist.sort_index(axis=0).sort_index(axis=1)
    #mask = mask.sort_index(axis=0).sort_index(axis=1)

    values = dist.values
    i, j = np.tril_indices_from(values, k=-1)
    pairwise_dists = pd.DataFrame.from_dict({'probe_id1':dist.index[i],
                                             'probe_id2': dist.columns[j],
                                             'distance': values[i, j]})

    pairwise_dists['gene1'] = pairwise_dists['probe_id1'].map(probe_ids)
    pairwise_dists['gene2'] = pairwise_dists['probe_id2'].map(probe_ids)
    pairwise_dists['same_gene'] = pairwise_dists['gene1'] == pairwise_dists['gene2']

    y_score = pairwise_dists.distance
    y_true = pairwise_dists.same_gene

    if metric == 'euclidean':
        auc = 1 - metrics.roc_auc_score(y_true, y_score)
    else:
        auc = metrics.roc_auc_score(y_true, y_score)

    return auc


def get_GO_presence_labels(genes_of_interest, min_GO_size=200, max_GO_size=300):
    """Creates a dataframe of GO-group presence for a list of genes.

    Args:
        genes_of_interest : must be iterable of entrez_gene_ids
        min_GO_size (int, optional): Min num of genes in GO group to be included. Defaults to 200.
        max_GO_size (int, optional): Max num of genes in GO group to be included. Defaults to 300.

    Returns:
        pd.DataFrame : df where index is entrezgene, columns are GO group with TRUE/FALSE presence values.
    """
    genes = pd.Series(genes_of_interest)
    go_group_presence = {}

    for GO in go2geneIDs:
        gene_ids = go2geneIDs[GO]
        
        # boolean vector (length is num of genes in embedding)
        in_go_group_vector = genes.isin(gene_ids)
        
        if (in_go_group_vector.sum() > min_GO_size) & (in_go_group_vector.sum() < max_GO_size):
            go_group_presence[GO] = in_go_group_vector


    result = pd.DataFrame(go_group_presence)
    result.index = genes
    result.index.name = 'entrezgene'


    return result


def filter_embedding_for_genes_in_GO(embedding, index_type='gene_symbol'):
    """Filters an embedding to only keep rows where genes have an annotation in GO.

    Args:
        embedding (pd.DataFrame): A DataFrame of shape (n_genes, n_dims)
        index_type (str, optional): Defaults to 'gene_symbol'.

    Returns:
        embedding (pd.DataFrame): A DataFrame of shape (n_genes, n_dims)
    """
    #gene_entrez_map = pd.read_csv( './data/raw/allen_human_fetal_brain/lmd_matrix_12566/rows_metadata.csv', usecols=['entrez_id', 'gene_symbol'])
    #gene_entrez_map = gene_entrez_map.dropna(subset=['entrez_id']).drop_duplicates(subset=['entrez_id'])

    gene_entrez_map = embedding.dropna(subset=['entrez_id']).drop_duplicates(subset=['entrez_id'])

    gene_entrez_map = gene_entrez_map[gene_entrez_map.entrez_id.isin(
        genes_in_GO)]

    """
    if index_type == 'gene_symbol':
        return embedding[embedding.index.isin(gene_entrez_map.gene_symbol)]
    else:
        return embedding[embedding.index.isin(gene_entrez_map.entrez_id)]
    """

    return gene_entrez_map


def merge_embedding_with_GO_labels(emb_df, GO_df):
    """Merges a gene_embedding with GO group presence df.

    Embedding cols are prefixed with emb_, while potential GO presence columns are prefixed with GO:

    Args:
        emb_df (pd.DataFrame): emb_df.index is gene_symbol
        GO_df (pd.DataFrame): GO_df.index is entrezgene

    Returns:
        (pd.DataFrame): Multi-index gene embedding with columns for GO presence concatenated.
    """
    # get df with gene_symbols and entrez_ids from fetal data (more updated than adult probes data)
    #all_genes = pd.read_csv('./data/raw/allen_human_fetal_brain/lmd_matrix_12566/rows_metadata.csv')
    #all_genes = all_genes[~((all_genes.gene_symbol.str.startswith('A_')) | (
        #all_genes.gene_symbol.str.startswith('CUST_')))].gene_symbol.drop_duplicates()

    #all_genes_w_entrez = utils.genesymbols_2_entrezids(all_genes)


    emb_df = emb_df.add_prefix('emb_')

    #df = emb_df.merge(all_genes_w_entrez, left_index=True,right_on='gene_symbol')

    emb_df = emb_df.rename(columns={"emb_gene_symbol": "gene_symbol", "emb_entrez_id": "entrez_id"})
    df = emb_df.merge(GO_df, left_on='entrez_id', right_index=True)

    return df.set_index(['entrez_id', 'gene_symbol'])


def perform_GOclass_eval(embedding_df,
                         rnd_state,
                         index_type='gene_symbol',
                         min_GO_size=200,
                         max_GO_size=300,
                         n_splits=5,
                         n_jobs=-1):

    if index_type == 'gene_symbol':
        embedding_df = filter_embedding_for_genes_in_GO(
            embedding_df, index_type='gene_symbol')


        #entrez_genelist = utils.genesymbols_2_entrezids(embedding_df.index)
        #GO_df = get_GO_presence_labels(genes_of_interest=entrez_genelist.entrez_id, min_GO_size=min_GO_size, max_GO_size=max_GO_size)

        emb_entrez_id = embedding_df['entrez_id']

        GO_df= get_GO_presence_labels(emb_entrez_id, min_GO_size=min_GO_size, max_GO_size=max_GO_size)

        gene_count_per_GO_group ={col: GO_df[col].sum() for col in GO_df.columns}



    elif index_type == 'entrez_id':
        embedding_df = filter_embedding_for_genes_in_GO(
            embedding_df, index_type='entrez_id')
        GO_df = get_GO_presence_labels(
            genes_of_interest=embedding_df.index, min_GO_size=min_GO_size, max_GO_size=max_GO_size)
    else:
        raise ValueError(
            "Error: specify index type as either 'gene_symbol' or 'entrez_id'.")

    # merge the embedding and GO_df to ensure they have same index
    # returns a multi-index df with gene_symbol and entrez_id
    merged_df = merge_embedding_with_GO_labels(emb_df=embedding_df, GO_df=GO_df)

    X = merged_df.loc[:, merged_df.columns.str.startswith('emb_')]
    y = merged_df.loc[:, merged_df.columns.str.startswith('GO:')]

    # ----

   
    # ----

    GO_SCORES = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=rnd_state)


    for GOlabel in y:

        #y_test_total = pd.Series([])
        #preds_total = []
        #probas_total = pd.DataFrame()

        f1_score_values = []
        auc_values = []

        print('--'*50)
        print(GOlabel)
        y_GO = y.loc[:, GOlabel]

        GO_term = goID2goTerm[GOlabel]
        GO_group_size = len(go2geneIDs[GOlabel])

        for i, (train_idx, test_idx) in enumerate(skf.split(X, y_GO)):

            model = LogisticRegression(penalty='none', n_jobs=n_jobs, random_state=rnd_state, max_iter=500)
            X_train = X.iloc[train_idx, :]
            y_train = y_GO.iloc[train_idx]
            X_test = X.iloc[test_idx, :]
            y_test = y_GO.iloc[test_idx]



            #model = make_pipeline(StandardScaler(), LogisticRegression(penalty='none', n_jobs=n_jobs))
            #model.fit(X_train, y_train)  # apply scaling on training data

            model.fit(X_train, y_train)

            # Extract predictions from fitted model
            preds = list(model.predict(X_test))
            # probs for classes ordered in same manner as model.classes_
            # model.classes_  >>  array([False,  True])
            probas = pd.DataFrame(model.predict_proba(
                X_test), columns=model.classes_)

            # Get metrics for each model
            f1 = f1_score(y_test, preds)
            auc = roc_auc_score(y_test, probas[True])

            f1_score_values.append(f1)
            auc_values.append(auc)

            #y_test_total = y_test_total.append(y_test)
            #preds_total += preds
            #probas_total = probas_total.append(probas)


            print("Fold")

        #preds_total = np.array(preds_total)

        #f1 = f1_score(y_test_total, preds_total)
        #auc = roc_auc_score(y_test_total, probas_total[True])

        f1 = np.mean(f1_score_values)
        auc = np.mean(auc_values)

        measures = {'GO_group': GOlabel,
                    'GO_group_title': GO_term,
                    'GO_group_size': GO_group_size,
                    'number of used genes':gene_count_per_GO_group[GOlabel],
                    'f1': f1,
                    'AUC': auc}



        GO_SCORES.append(measures)


    return pd.DataFrame(GO_SCORES)


def tmp_add_column(columns):
    #path = "/Users/pegah_abed/Documents/human_brains_final_files/final/final_files_and_folders/cortex_study"
    #path_to_embed = os.path.join(path,"plain_resnet/resnet50_embeddings_gene_level.csv" )

    path = "/Users/pegah_abed/Documents/human_brains_final_files/final/final_files_and_folders/cortex_study/random"
    path_to_embed = os.path.join(path, "random_all_training_embeddings_gene_level.csv")

    gene_level_embed_df = pd.read_csv(path_to_embed)
    print (list(gene_level_embed_df))

    path_to_info = "/Users/pegah_abed/Documents/human_brains_final_files/final/final_files_and_folders/cortex_study/human_ISH_info.csv"
    images_info = pd.read_csv(path_to_info)


    avail_cols = ['gene_symbol']
    for col in columns:
        if col not in list(images_info):
            print("column {} does not exist in image info file.".format(col))
        else:
            avail_cols.append(col)

    images_info = images_info[avail_cols]
    new_gene_level_embed_df = gene_level_embed_df.merge(images_info, how="left", on='gene_symbol')
    new_gene_level_embed_df = new_gene_level_embed_df.drop_duplicates(subset=['gene_symbol'])

    number_of_new_columns = len(avail_cols) - 1
    columns = list(new_gene_level_embed_df)
    columns = [columns[0]] + columns[(-1 * number_of_new_columns):] + columns[1:(-1 * number_of_new_columns)]

    new_image_level_embed_df = new_gene_level_embed_df[columns]
    new_image_level_embed_name = "random_all_training_embeddings_gene_level" + "_with_info.csv"
    new_image_level_embed_df_path = os.path.join(path, new_image_level_embed_name)

    new_image_level_embed_df.to_csv(new_image_level_embed_df_path, index=None)


def tmp_convert_to_gene_level():
    path = "/Users/pegah_abed/Documents/human_brains_final_files/final/final_files_and_folders/cortex_study/random"
    path_to_embed = os.path.join(path, "random_all_training_embeddings_image_level.csv")

    embeddings_file = pd.read_csv(path_to_embed)
    embeddings_file = embeddings_file.rename(columns={'id': 'image_id'})
    print (embeddings_file.head())

    path_to_info = "/Users/pegah_abed/Documents/human_brains_final_files/final/final_files_and_folders/cortex_study/human_ISH_info.csv"
    images_info = pd.read_csv(path_to_info)


    # perform left merge on the two dataframes to add gene_symbol to the embeddings.csv
    merged_df = embeddings_file.merge(images_info[["image_id", "gene_symbol"]], how="left", on="image_id")

    # reorder the dataframe columns
    merged_columns = list(merged_df)
    merged_columns = [merged_columns[0]] + [merged_columns[-1]] + merged_columns[1:-1]
    merged_df = merged_df[merged_columns]

    # drop the image_id column
    merged_df = merged_df.drop(columns=["image_id"])

    # group by gene_symbol and average over the embedding values
    grouped_df = merged_df.groupby(['gene_symbol']).mean()

    print(grouped_df.head())

    print("the number of genes is: {}".format(len(grouped_df)))

    # and then I want to save this file as gene_embddings in the same folder.
    item_name =  "random_all_training_embeddings_gene_level.csv"
    save_to_path = os.path.join(path, item_name)
    grouped_df.to_csv(save_to_path)


def get_patches_that_activate_neuron_the_most_and_the_least(ts, top_go_term, path_to_labels_file, path_to_gene_level_embeddings,
                                                            path_to_patch_level_embeddings):
    """
    First, get the donor level embeddings of the top gene (embeddings of the images that assay the top gene, aggregated to donor-level).
    This files is used as the data to get feature importance.

    Standardize the embeddings.

    Then, identify the most important feature by comparing the coefficients for a logistic regression model that has been mapped to the data.
    Next, for all the patches that assay the top gene, see which 2 patches have the highest value and which 2 patches have the lowest values
    for that features

    :param ts: str. Timestamp to indicate which set of embeddings to use.
    :param top_gene: str. The gene with the highest AUC score from the logistic regression binary classifier that predicts case vs control
    :param path_to_labels_file: str. Path to the labels csv file. Labels are True/False (case/control) on a donor level.
    :param path_to_patch_level_embeddings: str. Path to the patch level sz embeddings
    :return: None
    """
    path_to_per_gene_per_donor_file = os.path.join(sz_general_path, "per_gene_per_donor",
                                                   ts + "_embed_per_gene_per_donor", top_gene + ".csv")
    max_feature, max_score = feature_importance_with_lr(path_to_gene_level_embeddings, path_to_labels_file,
                                                        standardize=False)

    print('MAX')
    print(max_feature)
    path_to_info_file = os.path.join(sz_general_path, "human_ISH_info.csv")
    get_feature_value_from_patch_level(path_to_info_file, path_to_patch_level_embeddings, top_gene, max_feature)


def feature_importance_with_lr(path_to_embed_file, path_to_labels_file, standardize):
    if standardize == False:
        embed_df = pd.read_csv(path_to_embed_file)

    else:
        embed_df = pd.read_csv(path_to_embed_file)
        donor_id_col = embed_df['donor_id']
        embed_df = embed_df.drop(columns=['donor_id'])
        embed_df = embed_df.apply(lambda x: (x - x.mean()) / (x.std()), axis=1)
        embed_df['donor_id'] = donor_id_col

    label_df = pd.read_csv(path_to_labels_file)

    label_df = label_df.rename(columns={'ID': 'donor_id'})
    print(embed_df.head())
    print(label_df.head())

    left = embed_df
    right = label_df

    label_df = pd.merge(left, right, how='left', on='donor_id')[['donor_id', 'disease_diagnosis']]

    embed_df = embed_df.drop(columns=['donor_id'])
    label_df = label_df.drop(columns=['donor_id'])

    X = embed_df
    Y = label_df

    print(embed_df.head())
    print(label_df.head())

    model = LogisticRegression()
    # fit the model
    model.fit(X, Y)
    # get importance
    importance = model.coef_[0]
    # summarize feature importance

    max_score = math.inf * -1
    max_feature = None

    for i, v in enumerate(importance):
        print('Feature: {}, Score: {}'.format(i, abs(v)))
        if abs(v) > max_score:
            max_score = abs(v)
            max_feature = i

    print("----")
    print("Max:", max_feature, max_score)

    return (max_feature, max_score)

def create_gene_donor_one_hot_encoded_csv_file(path_to_info, path_to_save):

    info_df = pd.read_csv(path_to_info)



    genes= info_df['gene_symbol'].unique()
    donors = info_df['donor_id'].unique()

    print ("Number of unique genes: ", len(genes))
    print ("Number of unique donors: ", len(donors))

    tmp = np.zeros(dtype =int , shape=(len(genes), len(donors)))
    gene_donor_df = pd.DataFrame(tmp, index=genes, columns=donors)
    gene_donor_df.index.name = 'gene_symbol'
    print (gene_donor_df.head())



    gene_donor_map = {}

    group_by_gene = info_df.groupby('gene_symbol')
    for key, item in group_by_gene:
        gene_donor_map[key] = item['donor_id'].unique()


    for gene in gene_donor_map:
        for donor in gene_donor_map[gene]:
            gene_donor_df.loc[gene, donor] = 1

    print(gene_donor_df.head())

    # -------  add entrez_id ------

    gene_entrez_df = info_df[['gene_symbol', 'entrez_id']].drop_duplicates()
    gene_donor_df = gene_donor_df.merge(gene_entrez_df, how="left", on="gene_symbol")
    print (gene_donor_df.head())

    path_to_save = os.path.join(path_to_save, "cortex_gene_donor_one_hot_coded.csv")
    gene_donor_df.to_csv(path_to_save)
    return gene_donor_df






if __name__ == "__main__":
    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/cortex"

    path_to_info = os.path.join(general_path, "human_ISH_info.csv")
    path_to_save =  "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4/sz"
    gene_donor_one_hot_coded_df = create_gene_donor_one_hot_encoded_csv_file(path_to_info, path_to_save)



    #tmp_convert_to_gene_level()



    columns = ['entrez_id']
    #tmp_add_column(columns)



    general_path = "/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_3"

    ts_list = ["1603427156"] #["1603427490", "1603427156"]
    for ts in ts_list:
        embed_file_name = ts + "_triplet_all_training_embeddings_gene_level_with_info.csv"
        #path_to_embed = os.path.join(general_path, ts, embed_file_name)
        #path_to_embed = "/Users/pegah_abed/Documents/human_brains_final_files/final/final_files_and_folders/cortex_study/" \
         #               "plain_resnet/resnet50_embeddings_gene_level_with_info.csv"

        #path_to_embed = "/Users/pegah_abed/Documents/human_brains_final_files/final/final_files_and_folders/cortex_study/" \
          #             "random/random_all_training_embeddings_gene_level_with_info.csv"


        #embed_df = pd.read_csv(path_to_embed)

        embed_df =  gene_donor_one_hot_coded_df

        print(list(embed_df))



        min_GO_size = 40
        max_GO_size = 200
        rnd_state = 1
        go_scores = perform_GOclass_eval(embed_df,
                             rnd_state,
                             index_type='gene_symbol',
                             min_GO_size=min_GO_size,
                             max_GO_size=max_GO_size,
                             n_splits=5,
                             n_jobs=-1)


        go_scores = go_scores.sort_values(by=['AUC'], ascending=False)
        go_scores = go_scores.reset_index(drop=True)
        print (len(go_scores))
        print (np.mean(go_scores['AUC']))
        print ("*"*50)


        save_to_path = os.path.join("/Users/pegah_abed/Documents/old_Human_ISH/after_segmentation/dummy_4", "sz")

        #go_scores.to_csv(os.path.join(save_to_path, ts +"_avg_across_folds_go_scores_" + str(min_GO_size) + "_"
                                     # + str(max_GO_size) +  "_rand_state_" + str(rnd_state) +".csv"))

        #go_scores.to_csv(os.path.join(save_to_path, ts + "_avg_across_folds_go_scores_resnet_" +
         #              str(min_GO_size) + "_" + str(max_GO_size) +  "_rand_state_" + str(rnd_state) +".csv"))

       # go_scores.to_csv(os.path.join(save_to_path, ts + "_avg_across_folds_go_scores_random_" +
       #               str(min_GO_size) + "_" + str(max_GO_size) +  "_rand_state_" + str(rnd_state) +".csv"))

        go_scores.to_csv(os.path.join(save_to_path, ts +"_avg_across_folds_go_scores_gene_donor_one_hot_coded_" + str(min_GO_size) + "_"
         + str(max_GO_size) +  "_rand_state_" + str(rnd_state) +".csv"))




        

        


