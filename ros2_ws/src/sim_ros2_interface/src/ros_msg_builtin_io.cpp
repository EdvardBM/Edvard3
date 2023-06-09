#include <ros_msg_builtin_io.h>
#include <simLib.h>
#include <iostream>
#include <stubs.h>

ReadOptions::ReadOptions()
    : uint8array_as_string(false)
{
}

WriteOptions::WriteOptions()
    : uint8array_as_string(false)
{
}

void read__bool(int stack, bool *value, const ReadOptions *opt)
{
    simBool v;
    if(simGetStackBoolValueE(stack, &v) == 1)
    {
        *value = v;
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected bool");
    }
}

void read__byte(int stack, uint8_t *value, const ReadOptions *opt)
{
    simInt v;
    if(simGetStackInt32ValueE(stack, &v) == 1)
    {
        *value = (uint8_t)v;
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected bool");
    }
}

void read__char(int stack, unsigned char *value, const ReadOptions *opt)
{
    simInt v;
    if(simGetStackInt32ValueE(stack, &v) == 1)
    {
        *value = (char)v;
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected bool");
    }
}

void read__int8(int stack, int8_t *value, const ReadOptions *opt)
{
    simInt v;
    if(simGetStackInt32ValueE(stack, &v) == 1)
    {
        *value = v;
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected integer");
    }
}

void read__uint8(int stack, uint8_t *value, const ReadOptions *opt)
{
    simInt v;
    if(simGetStackInt32ValueE(stack, &v) == 1)
    {
        *value = v;
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected integer");
    }
}

void read__int16(int stack, int16_t *value, const ReadOptions *opt)
{
    simInt v;
    if(simGetStackInt32ValueE(stack, &v) == 1)
    {
        *value = v;
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected integer");
    }
}

void read__uint16(int stack, uint16_t *value, const ReadOptions *opt)
{
    simInt v;
    if(simGetStackInt32ValueE(stack, &v) == 1)
    {
        *value = v;
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected integer");
    }
}

void read__int32(int stack, int32_t *value, const ReadOptions *opt)
{
    simInt v;
    if(simGetStackInt32ValueE(stack, &v) == 1)
    {
        *value = v;
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected integer");
    }
}

void read__uint32(int stack, uint32_t *value, const ReadOptions *opt)
{
    simInt v;
    if(simGetStackInt32ValueE(stack, &v) == 1)
    {
        *value = v;
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected integer");
    }
}

void read__int64(int stack, int64_t *value, const ReadOptions *opt)
{
    // XXX: we represent Int64 as double - possible loss of precision!
    simDouble v;
    if(simGetStackDoubleValueE(stack, &v) == 1)
    {
        *value = v;
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected double");
    }
}

void read__uint64(int stack, uint64_t *value, const ReadOptions *opt)
{
    // XXX: we represent UInt64 as double - possible loss of precision!
    simDouble v;
    if(simGetStackDoubleValueE(stack, &v) == 1)
    {
        *value = v;
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected double");
    }
}

void read__float32(int stack, float *value, const ReadOptions *opt)
{
    simFloat v;
    if(simGetStackFloatValueE(stack, &v) == 1)
    {
        *value = v;
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected float");
    }
}

void read__float64(int stack, double *value, const ReadOptions *opt)
{
    simDouble v;
    if(simGetStackDoubleValueE(stack, &v) == 1)
    {
        *value = v;
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected double");
    }
}

void read__string(int stack, std::string *value, const ReadOptions *opt)
{
    simChar *str;
    simInt strSize;
    if((str = simGetStackStringValueE(stack, &strSize)) != NULL && strSize > 0)
    {
        *value = std::string(str, strSize);
        simPopStackItemE(stack, 1);
        simReleaseBufferE(str);
    }
    else
    {
        throw exception("expected string");
    }
}

void read__time(int stack, rclcpp::Time *value, const ReadOptions *opt)
{
    simDouble v;
    if(simGetStackDoubleValueE(stack, &v) == 1)
    {
        *value = rclcpp::Time(v);
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected double");
    }
}

void read__duration(int stack, rclcpp::Duration *value, const ReadOptions *opt)
{
    simDouble v;
    if(simGetStackDoubleValueE(stack, &v) == 1)
    {
        *value = rclcpp::Duration(v);
        simPopStackItemE(stack, 1);
    }
    else
    {
        throw exception("expected double");
    }
}

void write__bool(bool value, int stack, const WriteOptions *opt)
{
    simBool v = value;
    simPushBoolOntoStackE(stack, v);
}

void write__byte(uint8_t value, int stack, const WriteOptions *opt)
{
    simInt v = value;
    simPushInt32OntoStackE(stack, v);
}

void write__char(unsigned char value, int stack, const WriteOptions *opt)
{
    simInt v = value;
    simPushInt32OntoStackE(stack, v);
}

void write__int8(int8_t value, int stack, const WriteOptions *opt)
{
    simInt v = value;
    simPushInt32OntoStackE(stack, v);
}

void write__uint8(uint8_t value, int stack, const WriteOptions *opt)
{
    simInt v = value;
    simPushInt32OntoStackE(stack, v);
}

void write__int16(int16_t value, int stack, const WriteOptions *opt)
{
    simInt v = value;
    simPushInt32OntoStackE(stack, v);
}

void write__uint16(uint16_t value, int stack, const WriteOptions *opt)
{
    simInt v = value;
    simPushInt32OntoStackE(stack, v);
}

void write__int32(int32_t value, int stack, const WriteOptions *opt)
{
    simInt v = value;
    simPushInt32OntoStackE(stack, v);
}

void write__uint32(uint32_t value, int stack, const WriteOptions *opt)
{
    simInt v = value;
    simPushInt32OntoStackE(stack, v);
}

void write__int64(int64_t value, int stack, const WriteOptions *opt)
{
    // XXX: we represent Int64 as double - possible loss of precision!
    simDouble v = value;
    simPushDoubleOntoStackE(stack, v);
}

void write__uint64(uint64_t value, int stack, const WriteOptions *opt)
{
    // XXX: we represent UInt64 as double - possible loss of precision!
    simDouble v = value;
    simPushDoubleOntoStackE(stack, v);
}

void write__float32(float value, int stack, const WriteOptions *opt)
{
    simFloat v = value;
    simPushFloatOntoStackE(stack, v);
}

void write__float64(double value, int stack, const WriteOptions *opt)
{
    simDouble v = value;
    simPushDoubleOntoStackE(stack, v);
}

void write__string(std::string value, int stack, const WriteOptions *opt)
{
    const simChar *v = value.c_str();
    simPushStringOntoStackE(stack, v, value.length());
}

void write__time(rclcpp::Time value, int stack, const WriteOptions *opt)
{
    simDouble v = value.seconds();
    simPushDoubleOntoStackE(stack, v);
}

void write__duration(rclcpp::Duration value, int stack, const WriteOptions *opt)
{
    simDouble v = value.seconds();
    simPushDoubleOntoStackE(stack, v);
}

std::string goalUUIDtoString(const rclcpp_action::GoalUUID &uuid)
{
    static char n[17] = "0123456789abcdef";
    std::stringstream ss;
    for(size_t i = 0; i < UUID_SIZE; i++)
    {
        int h = (uuid[i] >> 4) & 0x0F;
        int l = (uuid[i] >> 0) & 0x0F;
        ss << n[h] << n[l];
    }
    return ss.str();
}

rclcpp_action::GoalUUID goalUUIDfromString(const std::string &uuidStr)
{
    static int val[256] = {
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4,5,6,7,8,9,0,0,0,0,0,0,
        0,10,11,12,13,14,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,10,11,12,13,14,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    };

    if(uuidStr.size() != UUID_SIZE * 2)
        log(sim_verbosity_warnings, boost::format("uuid '%s' has not the correct length (%d bytes)") % uuidStr % UUID_SIZE);

    rclcpp_action::GoalUUID ret;
    for(size_t i = 0, j = 0; i < (uuidStr.size() & ~1); i += 2, j++)
        ret[j] = (val[uuidStr[i]] << 4) | val[uuidStr[i+1]];
    return ret;
}

