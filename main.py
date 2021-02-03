# -*- coding: utf-8 -*-

from recommendation_filters import content_based_filter as cb, popularity_filter as pr

"""
import Sentiment_Analysis as sa;
import Feature_Genration as fg;
import Merge_Data as mg;
import eda as eda

Calling Following Functions Will Create and Store Required Data Files
sa.SA()
eda.Visualize_Data()
mg.final_data()
fg.genrate_Feature()

"""
print("On The Basis Of Popularity : ")
print(pr.get_popularity_recommend())

print("On The Basis of Content-Based : ")
print(cb.content_based_recommendation())

