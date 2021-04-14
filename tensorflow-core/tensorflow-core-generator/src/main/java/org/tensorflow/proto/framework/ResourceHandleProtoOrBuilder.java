// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/framework/resource_handle.proto

package org.tensorflow.proto.framework;

public interface ResourceHandleProtoOrBuilder extends
    // @@protoc_insertion_point(interface_extends:tensorflow.ResourceHandleProto)
    com.google.protobuf.MessageOrBuilder {

  /**
   * <pre>
   * Unique name for the device containing the resource.
   * </pre>
   *
   * <code>string device = 1;</code>
   */
  String getDevice();

  /**
   * <pre>
   * Unique name for the device containing the resource.
   * </pre>
   *
   * <code>string device = 1;</code>
   */
  com.google.protobuf.ByteString
  getDeviceBytes();

  /**
   * <pre>
   * Container in which this resource is placed.
   * </pre>
   *
   * <code>string container = 2;</code>
   */
  String getContainer();

  /**
   * <pre>
   * Container in which this resource is placed.
   * </pre>
   *
   * <code>string container = 2;</code>
   */
  com.google.protobuf.ByteString
  getContainerBytes();

  /**
   * <pre>
   * Unique name of this resource.
   * </pre>
   *
   * <code>string name = 3;</code>
   */
  String getName();

  /**
   * <pre>
   * Unique name of this resource.
   * </pre>
   *
   * <code>string name = 3;</code>
   */
  com.google.protobuf.ByteString
  getNameBytes();

  /**
   * <pre>
   * Hash code for the type of the resource. Is only valid in the same device
   * and in the same execution.
   * </pre>
   *
   * <code>uint64 hash_code = 4;</code>
   */
  long getHashCode();

  /**
   * <pre>
   * For debug-only, the name of the type pointed to by this handle, if
   * available.
   * </pre>
   *
   * <code>string maybe_type_name = 5;</code>
   */
  String getMaybeTypeName();

  /**
   * <pre>
   * For debug-only, the name of the type pointed to by this handle, if
   * available.
   * </pre>
   *
   * <code>string maybe_type_name = 5;</code>
   */
  com.google.protobuf.ByteString
  getMaybeTypeNameBytes();

  /**
   * <pre>
   * Data types and shapes for the underlying resource.
   * </pre>
   *
   * <code>repeated .tensorflow.ResourceHandleProto.DtypeAndShape dtypes_and_shapes = 6;</code>
   */
  java.util.List<ResourceHandleProto.DtypeAndShape>
  getDtypesAndShapesList();

  /**
   * <pre>
   * Data types and shapes for the underlying resource.
   * </pre>
   *
   * <code>repeated .tensorflow.ResourceHandleProto.DtypeAndShape dtypes_and_shapes = 6;</code>
   */
  ResourceHandleProto.DtypeAndShape getDtypesAndShapes(int index);

  /**
   * <pre>
   * Data types and shapes for the underlying resource.
   * </pre>
   *
   * <code>repeated .tensorflow.ResourceHandleProto.DtypeAndShape dtypes_and_shapes = 6;</code>
   */
  int getDtypesAndShapesCount();
  /**
   * <pre>
   * Data types and shapes for the underlying resource.
   * </pre>
   *
   * <code>repeated .tensorflow.ResourceHandleProto.DtypeAndShape dtypes_and_shapes = 6;</code>
   */
  java.util.List<? extends ResourceHandleProto.DtypeAndShapeOrBuilder>
  getDtypesAndShapesOrBuilderList();

  /**
   * <pre>
   * Data types and shapes for the underlying resource.
   * </pre>
   *
   * <code>repeated .tensorflow.ResourceHandleProto.DtypeAndShape dtypes_and_shapes = 6;</code>
   */
  ResourceHandleProto.DtypeAndShapeOrBuilder getDtypesAndShapesOrBuilder(
      int index);
}
