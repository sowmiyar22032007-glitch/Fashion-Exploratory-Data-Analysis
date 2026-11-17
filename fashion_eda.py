import numpy as np, pandas as pd, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
OUT = "."
# ---------- 1. Illustrative dataset (simulated; mirrors structure of public fashion datasets) ----------
cats = ["Athleisure","Sustainable Wear","Streetwear","Ethnic Wear","Formal Wear","Y2K Revival"]
months = pd.date_range("2023-01-01","2025-12-01",freq="MS")
t = np.arange(len(months))
base = {"Athleisure":(60,0.9,3),"Sustainable Wear":(30,1.6,2),"Streetwear":(55,0.6,6),
        "Ethnic Wear":(50,0.2,14),"Formal Wear":(52,-0.15,4),"Y2K Revival":(20,1.1,5)}
rows=[]
for c,(b,slope,amp) in base.items():
    season = amp*np.sin(2*np.pi*(t-8)/12)
    if c=="Ethnic Wear":  # festive peak Oct-Nov
        season = amp*np.exp(-((months.month.values-10.5)**2)/4)*1.2
    y = b+slope*t+season+rng.normal(0,2.2,len(t))
    for d,v in zip(months,y): rows.append((d,c,max(v,1)))
trend = pd.DataFrame(rows,columns=["month","category","search_index"])
trend_p = trend.pivot(index="month",columns="category",values="search_index")

n=1200
cat = rng.choice(cats,n,p=[.2,.14,.18,.18,.17,.13])
price_base={"Athleisure":1800,"Sustainable Wear":2600,"Streetwear":2200,"Ethnic Wear":3200,"Formal Wear":3800,"Y2K Revival":1500}
price = np.array([rng.lognormal(np.log(price_base[c]),0.35) for c in cat]).round(0)
# inject a few extreme outliers
idx = rng.choice(n,12,replace=False); price[idx]*=rng.uniform(3,5,12)
price=price.round(0)
discount = np.clip(rng.normal(22,10,n),0,60).round(0)
rating = np.clip(rng.normal(4.0,0.5,n)+ (np.array([c=="Sustainable Wear" for c in cat])*0.2),1,5).round(1)
season = rng.choice(["Spring/Summer","Autumn/Winter","Festive"],n,p=[.4,.4,.2])
material = rng.choice(["Cotton","Polyester","Linen","Silk","Recycled Fibre","Denim"],n,p=[.3,.22,.1,.08,.12,.18])
social = np.clip(rng.lognormal(6,0.8,n),50,None).round(0)
units = np.clip(20+0.012*social+0.9*discount+8*(rating-4)-0.004*price+rng.normal(0,12,n),1,None).round(0)
returns = np.clip(rng.normal(14,5,n)-2.5*(rating-4)+0.05*discount,0,60).round(1)
df = pd.DataFrame(dict(category=cat,season=season,material=material,price_inr=price,discount_pct=discount,
                       rating=rating,social_mentions=social,units_sold=units,return_rate_pct=returns))
# inject missing values + duplicates (for realistic cleaning step)
df.loc[rng.choice(n,45,replace=False),"rating"]=np.nan
df.loc[rng.choice(n,30,replace=False),"material"]=np.nan
df = pd.concat([df,df.sample(18,random_state=1)],ignore_index=True)
raw_shape = df.shape
miss = df.isna().sum().to_dict(); dups=int(df.duplicated().sum())
df.to_csv("fashion_illustrative_raw.csv",index=False)

# ---------- 2. Cleaning ----------
clean = df.drop_duplicates().copy()
clean["rating"]=clean["rating"].fillna(clean.groupby("category")["rating"].transform("median"))
clean["material"]=clean["material"].fillna(clean["material"].mode()[0])
Q1,Q3=clean.price_inr.quantile([.25,.75]); IQR=Q3-Q1
hi=Q3+1.5*IQR; out_n=int((clean.price_inr>hi).sum())
clean["price_capped"]=clean.price_inr.clip(upper=hi)
clean.to_csv("fashion_illustrative_clean.csv",index=False)

desc = clean[["price_inr","discount_pct","rating","social_mentions","units_sold","return_rate_pct"]].describe().T.round(2)
desc.to_csv("summary_stats.csv")
corr = clean[["price_capped","discount_pct","rating","social_mentions","units_sold","return_rate_pct"]].corr().round(2)

# ---------- 3. Charts ----------
plt.rcParams.update({"font.family":"DejaVu Sans","axes.spines.top":False,"axes.spines.right":False,"font.size":10})
pal=["#1f4e79","#2e8b57","#d9822b","#8e44ad","#c0392b","#7f8c8d"]

fig,ax=plt.subplots(figsize=(8,4.2))
for c,col in zip(cats,pal): ax.plot(trend_p.index,trend_p[c],label=c,color=col,lw=2)
ax.set_title("Fig 1. Monthly search-interest index by fashion category (2023-2025)",fontsize=11,loc="left")
ax.set_ylabel("Search interest index"); ax.legend(ncol=3,fontsize=8,frameon=False,loc="upper left")
ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig("fig1_trend.png",dpi=200); plt.close()

rev = clean.assign(rev=clean.units_sold*clean.price_capped*(1-clean.discount_pct/100)).groupby("category").rev.sum().sort_values()
fig,ax=plt.subplots(figsize=(8,3.8))
ax.barh(rev.index,rev.values/1e6,color="#1f4e79")
for i,v in enumerate(rev.values/1e6): ax.text(v+0.05,i,f"{v:.1f}M",va="center",fontsize=9)
ax.set_title("Fig 2. Estimated net revenue by category (INR millions)",fontsize=11,loc="left")
ax.set_xlabel("INR millions"); fig.tight_layout(); fig.savefig("fig2_bar.png",dpi=200); plt.close()

mat=clean.material.value_counts()
fig,ax=plt.subplots(figsize=(6,4.2))
ax.pie(mat.values,labels=mat.index,autopct="%1.0f%%",startangle=90,colors=["#1f4e79","#2e8b57","#d9822b","#8e44ad","#c0392b","#7f8c8d"],wedgeprops=dict(width=.55,edgecolor="white"),textprops=dict(fontsize=9))
ax.set_title("Fig 3. Share of products by material",fontsize=11,loc="left")
fig.tight_layout(); fig.savefig("fig3_pie.png",dpi=200); plt.close()

fig,ax=plt.subplots(figsize=(6.6,4.8))
im=ax.imshow(corr.values,cmap="RdBu_r",vmin=-1,vmax=1)
lab=["Price","Discount","Rating","Social\nmentions","Units\nsold","Return\nrate"]
ax.set_xticks(range(6)); ax.set_yticks(range(6)); ax.set_xticklabels(lab,fontsize=8); ax.set_yticklabels(lab,fontsize=8)
for i in range(6):
    for j in range(6): ax.text(j,i,f"{corr.values[i,j]:.2f}",ha="center",va="center",fontsize=8,color="white" if abs(corr.values[i,j])>.5 else "black")
ax.set_title("Fig 4. Correlation heatmap",fontsize=11,loc="left"); fig.colorbar(im,shrink=.8)
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig("fig4_heatmap.png",dpi=200); plt.close()

order=clean.groupby("category").price_inr.median().sort_values().index
fig,ax=plt.subplots(figsize=(8,4.2))
ax.boxplot([clean[clean.category==c].price_inr for c in order],tick_labels=list(order),flierprops=dict(marker="o",markersize=3,markerfacecolor="#c0392b",markeredgecolor="#c0392b"),boxprops=dict(color="#1f4e79"),medianprops=dict(color="#d9822b",lw=2))
ax.set_title("Fig 5. Price distribution and outliers by category (INR)",fontsize=11,loc="left")
ax.set_ylabel("Price (INR)"); plt.setp(ax.get_xticklabels(),rotation=15,fontsize=8); ax.grid(axis="y",alpha=.25)
fig.tight_layout(); fig.savefig("fig5_box.png",dpi=200); plt.close()

fig,ax=plt.subplots(figsize=(6.6,4.2))
ax.scatter(clean.discount_pct,clean.units_sold,s=9,alpha=.4,color="#1f4e79")
m,b=np.polyfit(clean.discount_pct,clean.units_sold,1); xs=np.linspace(0,60,50); ax.plot(xs,m*xs+b,color="#d9822b",lw=2)
ax.set_title("Fig 6. Discount vs units sold",fontsize=11,loc="left"); ax.set_xlabel("Discount (%)"); ax.set_ylabel("Units sold"); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig("fig6_scatter.png",dpi=200); plt.close()

# ---------- 4. Facts for the report ----------
g0=trend_p.iloc[:12].mean(); g2=trend_p.iloc[-12:].mean(); growth=((g2/g0-1)*100).round(0)
peak_month = {c:int(trend_p[c].groupby(trend_p.index.month).mean().idxmax()) for c in cats}
sust_ret = clean.groupby("category").return_rate_pct.mean().round(1).to_dict()
facts=dict(raw_rows=raw_shape[0],raw_cols=raw_shape[1],miss=miss,dups=dups,clean_rows=len(clean),out_n=out_n,hi=float(hi),
  growth=growth.to_dict(),peak_month=peak_month,rev=(rev/1e6).round(1).to_dict(),corr=corr.to_dict(),
  mat=mat.to_dict(),ret=sust_ret,slope=float(m),
  med_price=clean.groupby("category").price_inr.median().round(0).to_dict(),
  rating_by_cat=clean.groupby("category").rating.mean().round(2).to_dict(),
  season_units=clean.groupby("season").units_sold.mean().round(1).to_dict(),
  desc=desc.reset_index().to_dict(orient="records"))
json.dump(facts,open("facts.json","w"),indent=1,default=float)
print(json.dumps({k:v for k,v in facts.items() if k!="desc"},indent=1,default=float))
print(desc)
