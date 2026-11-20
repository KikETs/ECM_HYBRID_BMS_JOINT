#include <math.h>

static double ew(double state, double x, double dt, double tau)
{
    return state + (1.0-exp(-dt/tau)) * (x-state);
}

/* Input columns: time, current, voltage, T, SOC, SOH, R0, R1, tau1,
   R2, tau2, OCV, hysteresis. Output: twelve primary and nine extra features. */
int snapshot(int count, double *data, float *features)
{
    const double eps = 1e-9;
    const double variant_tau[5] = {2.,4.,8.,16.,32.};
    double next = data[0]+600., v1=0., v2=0., h=0.;
    double is=0., ir=0., ii=0., sr=0., ss=0., hi=0., rest=0., duty=0., i2=0., et=25.;
    double vis[5]={0}, vsr[5]={0}, vss[5]={0};
    double lir=0., lii=0., lsr=0., lss=0., gir=0., gii=0., gsr=0., gss=0., load=0.;
    int rows=0;
    for (int k=1; k<count; ++k) {
        double *s=data+k*13;
        double dt=s[0]-(s-13)[0];
        if (!(isfinite(dt) && dt>0.)) continue;
        if (dt>60.) dt=60.;
        double I=s[1], ia=fabs(I), V=s[2], T=s[3];
        double r0=s[6], r1=s[7], t1=s[8], r2=s[9], t2=s[10];
        double r=V-(s[11]+s[12]*h+I*r0+v1+v2);
        is=ew(is,I,dt,8.);
        ir=ew(ir,I*r,dt,600.); ii=ew(ii,I*I,dt,600.);
        sr=ew(sr,is*r,dt,600.); ss=ew(ss,is*is,dt,600.);
        double over=fmax(0.,ia-10.);
        hi=ew(hi,over*over,dt,600.); rest=ew(rest,ia<.5,dt,600.);
        duty=ew(duty,ia>5.,dt,300.); i2=ew(i2,I*I,dt,600.); et=ew(et,T,dt,600.);
        for(int n=0;n<5;++n){
            vis[n]=ew(vis[n],I,dt,variant_tau[n]);
            vsr[n]=ew(vsr[n],vis[n]*r,dt,600.);
            vss[n]=ew(vss[n],vis[n]*vis[n],dt,600.);
        }
        lir=ew(lir,I*r,dt,1800.); lii=ew(lii,I*I,dt,1800.);
        lsr=ew(lsr,is*r,dt,1800.); lss=ew(lss,is*is,dt,1800.);
        load=ia>5. ? load+dt : 0.;
        if(load>=3.){
            gir=ew(gir,I*r,dt,600.); gii=ew(gii,I*I,dt,600.);
            gsr=ew(gsr,is*r,dt,600.); gss=ew(gss,is*is,dt,600.);
        }
        double a1=exp(-dt/t1), a2=exp(-dt/t2);
        v1=v1*a1+r1*(1.-a1)*I; v2=v2*a2+r2*(1.-a2)*I;
        double ah=exp(-fabs(20.*I*dt/3600./3.));
        h=ah*h+(1.-ah)*((I>0.)-(I<0.));
        if(s[0]>=next){
            float *o=features+rows*21;
            o[0]=ir/(ii+eps)*1000.; o[1]=sr/(ss+eps)*1000.; o[2]=log10(ii+eps);
            o[3]=hi; o[4]=rest; o[5]=duty; o[6]=s[4]; o[7]=s[5]; o[8]=et;
            o[9]=sqrt(fmax(i2,0.)); o[10]=(r0+r1)*1000.; o[11]=r2*1000.;
            for (int n=0;n<5;++n) o[12+n]=vsr[n]/(vss[n]+eps)*1000.;
            o[17]=lir/(lii+eps)*1000.; o[18]=lsr/(lss+eps)*1000.;
            o[19]=gir/(gii+eps)*1000.; o[20]=gsr/(gss+eps)*1000.;
            next=s[0]+600.;
            ++rows;
        }
    }
    return rows;
}
