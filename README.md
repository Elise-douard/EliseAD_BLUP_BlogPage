# BLUP: Brain Learning Unicorn Project

### ***Can a model predict the genetic profile of an individual based on brain regions volumes?***

<ins>Author:</ins> Elise Alix Douard

<ins>Other project contributors:</ins>
BHS, Hannah Kiesow [@hannahmaykiesow](https://twitter.com/hannahmaykiesow) (also working on UKBiobank), Kuldeep Kumar [@meetkd007](https://twitter.com/meetkd007) (extracting data from UKBB servers)

This Github page is the final product for the week 3 deliverable related to the data visualization. 

# Background

<p align="center">
  <img src="blog_content/illustration_genetic.png">
</p>

<p> <font size="1">Source: Illustration inspired from freepik.com content and adapted on adobe illustrator</font></p> 

Copy number variants (CNVs) are a family of structural variation of the chromosomes. They can be either a gain or a loss of a chromosome portion in comparission to a genome of reference. 

Sometimes, CNVs can be pathogenic, meaning that they are formally associated to neurodevelopmental or psychiatric disorders, such as autism spectrum disorders (ASD), Schizophrenia (SZ) or intellectual disability (ID). 

Such pathogenic CNVs have been associated to significant alterations of brain volume (Modenato et al., ; [Martin-Brevet et al., 2018](http://www.sciencedirect.com/science/article/pii/S000632231831401X) ; [Maillard et al., 2015](https://www.nature.com/articles/mp2014145)) or connectivity ([Moreau et al., 2019](https://www.biorxiv.org/content/10.1101/862615v1.full)). 

Notably, there were common alterations of the insula volume when comparing structural brain alterations due to pathogenic CNVs and due to a neurodevelopmental disorder (e.g. ASD or SZ) (). 

# Aim of the project

<p align="center">
  <img src="blog_content/BLUPproject.png">
</p>

<p> <font size="1">Source: Illustration inspired from freepik.com content and adapted on adobe illustrator</font></p> 

- Main goal: 

This project aims to feed a machine learning model with brain volumes to predict if an individual is carrier of a potentially pathogenic CNV.

- Personal goals: 

- [x] Learn how to python
- [x] Learn how to machine learning
- [x] Learn how to interactive plot

# Methods

## Raw data
[a](blog_content/DesikanLeftParcellation_lh.html)



<p align="center">
<a href="blog_content/DesikanLeftParcellation_lh.html"><img src="blog_content/Desikan_lh_atlas.png" width="400" height="275" title="Click to access to the interactive 3D plot" alt="Flower"></a>
</p>

Here is the code used to create the 3D map of the left brain Desikan parcellation (34 regions). 

```python
# Import a template of Freesurfer processed left hemisphere based on Desikan atlas to create the ROI map
annotation_path_lh = "free_surfer_parcellation/lh.aparc.annot"
annotation_lh = read_annot(annotation_path_lh)

ROI_map_lh = annotation_lh[0]

fsaverage5 = datasets.fetch_surf_fsaverage(mesh='fsaverage', data_dir=None)
fig0_lh = plotting.view_surf(fsaverage5.infl_left, ROI_map_lh, cmap = "gist_ncar", symmetric_cmap=True,
                         colorbar = False, 
                         title = "Desikan atlas of the left hemisphere" )

```

## Final brain volumes