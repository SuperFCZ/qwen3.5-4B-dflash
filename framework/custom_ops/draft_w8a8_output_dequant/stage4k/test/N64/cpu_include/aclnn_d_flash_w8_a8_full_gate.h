#pragma once
struct aclTensor{void*data;};struct aclOpExecutor{};
aclTensor* aclCreateTensor(const int64_t*,uint64_t,int,const int64_t*,int64_t,int,const int64_t*,uint64_t,void*);
int aclDestroyTensor(const aclTensor*);
int aclnnDFlashW8A8FullGateGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,int64_t,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
int aclnnDFlashW8A8FullGate(void*,uint64_t,aclOpExecutor*,aclrtStream);
