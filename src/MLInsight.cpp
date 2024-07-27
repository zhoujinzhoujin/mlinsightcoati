/*
@author: Steven Tang <jtang@umass.edu>
*/
#include <pybind11/pybind11.h>
#include <iostream>
#include <map>
#include <cuda_runtime.h>
#include <cupti.h>
#include <pybind11/functional.h>

inline void cudaAssert(cudaError_t err, const char *__file, int __line) {
    if (cudaSuccess != err) {
        fprintf(stderr,"result != cudaSuccess %s %d %d\n", err != cudaSuccess ? "true" : "false", err, cudaSuccess);
        fprintf(stderr,"ERR: %s:%d  CUDA runtime error code=%d(%s) \"%s\" \n", __file, __line, err, cudaGetErrorName(err),
                cudaGetErrorString(err));
        exit(-1);
    }
}
// Kill program and print error message if a CUDA runtime API fails.
#define CUDA_ASSERT(result) {std::string srcFile=__FILE__; cudaAssert(result,srcFile.c_str(),__LINE__);}
//Kill program and print error message if a CUDA driverRecord API fails.
#define CU_ASSERT(result) {std::string srcFile=__FILE__; cuAssert(result,srcFile.c_str(),__LINE__);}

#define CUPTI_CALL(call)                                                          \
    do {                                                                          \
        CUptiResult _status = call;                                               \
        if (_status != CUPTI_SUCCESS) {                                           \
            const char *errstr;                                                   \
            cuptiGetResultString(_status, &errstr);                               \
            fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",  \
                    __FILE__, __LINE__, #call, errstr);                           \
            exit(EXIT_FAILURE);                                                    \
        }                                                                         \
    } while (0)


namespace mlinsight {
    static std::function<void(uint64_t,uint64_t)>* cuMemAllocCBPtr;
    static std::function<void(uint64_t)>* cuMemFreeCBPtr;

    void hello(){
        size_t free=0;
        size_t total=0;
        CUDA_ASSERT(cudaMemGetInfo(&free,&total));
        printf("If you save this message, then MLInsight is correctly installed.\n Free:%ld Total=%ld\n",free,total);
        //cuMemAllocFinishedCBPtr();
        //cuMemFreeFinishedCBPtr();
    }

    void cuptiAPICallBack(void *userdata, CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const void *cbdata) {
        assert(domain == CUPTI_CB_DOMAIN_DRIVER_API);
        
        //printf("Cupti Callback is invoked\n");
        
        const CUpti_CallbackData *cbInfo = (CUpti_CallbackData *) cbdata;
        if(cbInfo->callbackSite==CUPTI_API_EXIT){
            if (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAlloc) {
                auto* params=(cuMemAlloc_params_st*)cbInfo->functionParams;
                cuMemAllocCBPtr->operator()(*(params->dptr),params->bytesize);
                //fprintf(stderr,"cuMemAllocFinishedCBPtr\n");
            } else  if (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2) {
                auto* params=(cuMemAlloc_v2_params_st*)cbInfo->functionParams;
                cuMemAllocCBPtr->operator()(*(params->dptr),params->bytesize);
            } 
            else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemFree) {
                auto* params=(cuMemFree_params_st*)cbInfo->functionParams;
                cuMemFreeCBPtr->operator()(params->dptr);
            }
            else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2){
                auto* params=(cuMemFree_v2_params_st*)cbInfo->functionParams;
                cuMemFreeCBPtr->operator()(params->dptr);
            }
        }
    }


    void install(std::function<void(uint64_t,uint64_t)> cmMemAllocCB, std::function<void(uint64_t)> cmMemFreeCB){
        cuMemAllocCBPtr=new std::function<void(uint64_t,uint64_t)>(std::move(cmMemAllocCB));
        cuMemFreeCBPtr=new std::function<void(uint64_t)>(std::move(cmMemFreeCB));
         
        CUpti_SubscriberHandle subscriber;
        CUptiResult cuptiRet= cuptiSubscribe(&subscriber, (CUpti_CallbackFunc) cuptiAPICallBack, nullptr);
        if(cuptiRet == CUPTI_SUCCESS){
            CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));
            fprintf(stderr,"Cupti initialized\n");
        }else{
            fprintf(stderr,"Warning, cupti instialized failed. Cupti cross-checking will not be effective for process %zd", getpid());
        }
      
    }
}


PYBIND11_MODULE(_mlinsight, m) {
    m.doc() = "MLInsight Python Extension Version ";
    m.def("hello", &mlinsight::hello, "Prints MLInsight Version info");
    m.def("install", &mlinsight::install, "Prints MLInsight Version info");

    auto cleanup_callback = []() {
        // perform cleanup here -- this function is called with the GIL held
        delete mlinsight::cuMemAllocCBPtr;
        delete mlinsight::cuMemFreeCBPtr;
    };

    m.add_object("_cleanup", pybind11::capsule(cleanup_callback));

}