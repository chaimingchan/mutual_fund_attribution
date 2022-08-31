#!/usr/bin/env python
# coding: utf-8

# # 构建函数TM-HM-CL
# 
# 本代码计算基金择时选股能力，所用模型为TM-HM-CL模型。用户自选时间区间与基金，则调用本代码
# 
# 
# ## 参数设置
# 1、基金代码
# 2、对标指数代码
# 3、无风险利率
# 4、起始日期
# 5、结束日期

##默认参数：基金净值数据、指数收盘价数据

# In[1]:


import A_get_data_tmhmcl

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn import linear_model
import statsmodels.api as sm
import sqlalchemy as sql
engine = sql.create_engine('mysql+pymysql://root:123456@localhost:3306/Mutualfund')
import warnings
warnings.filterwarnings("ignore")


class AttributionModel_TmHmCl:
    
    
    def __init__(self,code,index_name,start_date,end_date,rf,nav_data=A_get_data_tmhmcl.Get_TMHMCL_Data_Csv.Get_Fund_Adjusted_Nav_Data_Csv(),index_data=A_get_data_tmhmcl.Get_TMHMCL_Data_Csv.Get_Stock_Index_Data_Csv()):
        self.code=code
        self.index_name=index_name
        self.start_date=start_date
        self.end_date=end_date
        self.rf=rf    
        self.nav_data=nav_data
        self.index_data=index_data
        
        '''
        输入参数：
        code: 基金代码
        index_name: 指数名称
        start_date：计算起始日期
        end_date：计算结束日期
        rf：无风险收益率（可为数字，可为序列）  
        nav_data:所有基金近三年净值数据
        index_data:常用宽基指数（也可以是其他指数）近三年
        '''


    def Mix_nav_index_data(self):
               
        list_nav=self.nav_data.loc[:,['Date',self.code]].set_index('Date')
        index_price=self.index_data.loc[:,['日期',self.index_name]].set_index('日期')
            
        #数据对齐
        mixnav=pd.merge(list_nav,index_price,left_index=True,right_index=True,how='inner').dropna()
        mixnav.index=pd.to_datetime(mixnav.index)
        mixnav=mixnav[pd.to_datetime(self.start_date):pd.to_datetime(self.end_date)]
        
        if len(mixnav)>50:
            
            list_value=mixnav.iloc[:,0]   
            list_index=mixnav.iloc[:,1]

            r_rf=pd.DataFrame(list_value).pct_change().dropna()-self.rf
            r_rf=pd.DataFrame(np.array(r_rf),columns=['r'])
            rm_rf=pd.DataFrame(list_index).pct_change().dropna()-self.rf
            rm_rf=pd.DataFrame(np.array(rm_rf),columns=['rm'])
            
            return r_rf,rm_rf,mixnav.index.min(),mixnav.index.max()#返回数据计算区间实际起始日期，结束日期
           
            
    def TM_Model(self):      
            
            # 计算TM模型
            if self.Mix_nav_index_data() is not None:
                
                y=np.array(self.Mix_nav_index_data()[0])
                xtm1=self.Mix_nav_index_data()[1]
                xtm2=xtm1**2
                xtm=pd.concat([xtm1,xtm2],axis=1 )
    
                xxtm=sm.add_constant(xtm)
                tm_model=sm.OLS(y,xxtm)
                tm_results=tm_model.fit()
                tm_a=tm_results.params[0]
                tm_beta1=tm_results.params[1]
                tm_beta2=tm_results.params[2]
                tm_p0=tm_results.pvalues[0]
                tm_p1=tm_results.pvalues[1]
                tm_p2=tm_results.pvalues[2]
                tm_r2=tm_results.rsquared
                out_tm=[self.code,self.Mix_nav_index_data()[2],self.Mix_nav_index_data()[3],tm_a,tm_beta1,tm_beta2,tm_p0,tm_p1,tm_p2,tm_r2]
                tm1=pd.DataFrame(out_tm,index=['基金代码','起始日期','结束日期','tm_a','tm_beta1','tm_beta2','tm_p0','tm_p1','tm_p2','tm_r2'])
                tm_result=pd.DataFrame(tm1.values.T,index=tm1.columns,columns=tm1.index)
            else:
                tm_result=None
           
            return tm_result


    def HM_Model(self):
            # 计算HM模型
            if self.Mix_nav_index_data() is not None:
                y=np.array(self.Mix_nav_index_data()[0])
                xtm1=self.Mix_nav_index_data()[1]
                xhm2=pd.DataFrame(np.maximum(np.array(xtm1), 0))
                xhm=pd.concat([xtm1,xhm2],axis=1 )
    
                xxhm=sm.add_constant(xhm)
                hm_model=sm.OLS(y,xxhm)
                hm_results=hm_model.fit()
                hm_a=hm_results.params[0]
                hm_beta1=hm_results.params.iloc[1]
                hm_beta2=hm_results.params.iloc[2]
                hm_p0=hm_results.pvalues[0]
                hm_p1=hm_results.pvalues.iloc[1]
                hm_p2=hm_results.pvalues.iloc[2]
                hm_r2=hm_results.rsquared
                out_hm=[self.code,self.Mix_nav_index_data()[2],self.Mix_nav_index_data()[3],hm_a,hm_beta1,hm_beta2,hm_p0,hm_p1,hm_p2,hm_r2]
                hm1=pd.DataFrame(out_hm,index=['基金代码','起始日期','结束日期','hm_a','hm_beta1','hm_beta2','hm_p0','hm_p1','hm_p2','hm_r2'])
                hm_result=pd.DataFrame(hm1.values.T,index=hm1.columns,columns=hm1.index)
            else:
                hm_result=None
            
            return hm_result
        
    def CL_Model(self):

            # 计算CL模型
            if self.Mix_nav_index_data() is not None:
                y=np.array(self.Mix_nav_index_data()[0])
                xtm1=self.Mix_nav_index_data()[1]
                xcl1=pd.DataFrame(np.maximum(np.array(xtm1), 0))
                xcl2=pd.DataFrame(np.minimum(np.array(xtm1), 0))
                xcl=pd.concat([xcl1,xcl2],axis=1 )  
    
                xxcl=sm.add_constant(xcl)
                cl_model=sm.OLS(y,xxcl)
                cl_results=cl_model.fit()
                cl_a=cl_results.params.iloc[0]
                cl_beta1=cl_results.params.iloc[1]
                cl_beta2=cl_results.params.iloc[2]
                cl_p0=cl_results.pvalues.iloc[0]
                cl_p1=cl_results.pvalues.iloc[1]
                cl_p2=cl_results.pvalues.iloc[2]
                cl_r2=cl_results.rsquared
                out_cl=[self.code,self.Mix_nav_index_data()[2],self.Mix_nav_index_data()[3],cl_a,cl_beta1,cl_beta2,cl_p0,cl_p1,cl_p2,cl_r2]
                cl1=pd.DataFrame(out_cl,index=['基金代码','起始日期','结束日期','cl_a','cl_beta1','cl_beta2','cl_p0','cl_p1','cl_p2','cl_r2'])
                cl_result=pd.DataFrame(cl1.values.T,index=cl1.columns,columns=cl1.index)
            else:
                cl_result=None

            return cl_result


# In[2]:


if __name__ == '__main__':

    code='000001.OF'
    #code='970166.OF'
    index_name='上证指数'
    start_date='20210701'
    end_date='20220701'
    rf=1.5/36500
    
    res=AttributionModel_TmHmCl(code,index_name,start_date,end_date,rf).TM_Model()

    print(res)






