# -*- coding: utf-8 -*-

import Pop_Rec as pr;
import Content_Based as cb


"""
import Sentiment_Analysis as sa;
import Feature_Genration as fg;
import Merge_Data as mg;
import eda as eda

Calling Following Functions Will Create and Store Required Data Files
eda.Visualize_Data()
sa.SA()
mg.final_data()
fg.genrate_Feature()

"""
print("On The Basis Of popularity : ")
print(pr.get_popularity_recommend())

print("On The Basis of Content-Based : ")
print(cb.content_based_recommendation())

#print("On The Basis of Collaborative-Filtering : ")
#print("")
