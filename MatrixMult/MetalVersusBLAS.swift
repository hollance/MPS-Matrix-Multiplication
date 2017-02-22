import Foundation
import Accelerate
import MetalPerformanceShaders

// Sizes of the matrices: C = A x B.
private let rowsA = 5000
private let columnsA = 2000
private let rowsB = columnsA
private let columnsB = 3000
private let rowsC = rowsA
private let columnsC = columnsB

class MetalVersusBLAS {
  private var device: MTLDevice!
  private var commandQueue: MTLCommandQueue!

  private var matrixMultiplication: MPSMatrixMultiplication!
  private var matrixA: MPSMatrix!
  private var matrixB: MPSMatrix!
  private var matrixC: MPSMatrix!

  private var arrayA = [Float](repeating: 0, count: rowsA * columnsA)
  private var arrayB = [Float](repeating: 0, count: rowsB * columnsB)
  private var arrayC = [Float](repeating: 0, count: rowsC * columnsC)

  func run(logger: Logger) {
    logger.log(message: "*** Metal vs. BLAS matrix multiplication ***")
    logger.log(message: "(\(rowsA) x \(columnsA)) * (\(rowsB) x \(columnsB)) = (\(rowsC) x \(columnsC))")

    randomizeArrays()
    initMPS()
    predictMPS(logger: logger)
    predictBLAS(logger: logger)
    compareResults(logger: logger)

    logger.log(message: "")
  }

  private func randomizeArrays() {
    // Fill up A and B with random floating point numbers (between -1 and +1).
    for i in 0..<arrayA.count {
      arrayA[i] = Float(2*drand48() - 1)
    }
    for i in 0..<arrayB.count {
      arrayB[i] = Float(2*drand48() - 1)
    }
  }

  private func initMPS() {
    device = MTLCreateSystemDefaultDevice()
    guard device != nil else {
      fatalError("Error: This device does not support Metal")
    }

    guard MPSSupportsMTLDevice(device) else {
      fatalError("Error: This device does not support Metal Performance Shaders")
    }

    commandQueue = device.makeCommandQueue()

    matrixMultiplication = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: rowsC, resultColumns: columnsC, interiorColumns: columnsA, alpha: 1, beta: 0)

    // For optimal speed, we should use the recommended row stride.
    //let rowBytesA = MPSMatrixDescriptor.rowBytes(fromColumns: columnsA, dataType: .float32)
    //print("preferred stride \(rowBytesA), my stride \(columnsA * MemoryLayout<Float>.stride)")

    // The contents of the arrays are copied into the MTLBuffers. Note that we
    // don't copy arrayC into bufferC because it's just zeros (arrayC is only 
    // used to store the results of the BLAS matrix multiply).
    let bufferA = device.makeBuffer(bytes: arrayA, length: rowsA * columnsA * MemoryLayout<Float>.stride, options: [])
    let bufferB = device.makeBuffer(bytes: arrayB, length: rowsB * columnsB * MemoryLayout<Float>.stride, options: [])
    let bufferC = device.makeBuffer(length: rowsC * columnsC * MemoryLayout<Float>.stride, options: [])

    let descA = MPSMatrixDescriptor(dimensions: rowsA, columns: columnsA, rowBytes: columnsA * MemoryLayout<Float>.stride, dataType: .float32)
    let descB = MPSMatrixDescriptor(dimensions: rowsB, columns: columnsB, rowBytes: columnsB * MemoryLayout<Float>.stride, dataType: .float32)
    let descC = MPSMatrixDescriptor(dimensions: rowsC, columns: columnsC, rowBytes: columnsC * MemoryLayout<Float>.stride, dataType: .float32)

    matrixA = MPSMatrix(buffer: bufferA, descriptor: descA)
    matrixB = MPSMatrix(buffer: bufferB, descriptor: descB)
    matrixC = MPSMatrix(buffer: bufferC, descriptor: descC)
  }

  private func predictMPS(logger: Logger) {
    let elapsed = timeIt {
      let commandBuffer = commandQueue.makeCommandBuffer()

      matrixMultiplication.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixB, resultMatrix: matrixC)

      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    }
    logger.log(message: "MPS took \(elapsed) seconds")
  }

  private func predictBLAS(logger: Logger) {
    let elapsed = timeIt {
      arrayA.withUnsafeBufferPointer { ptrA in
        arrayB.withUnsafeBufferPointer { ptrB in
          arrayC.withUnsafeMutableBufferPointer { ptrC in
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(rowsA), Int32(columnsB), Int32(columnsA), 1, ptrA.baseAddress, Int32(columnsA), ptrB.baseAddress, Int32(columnsB), 0, ptrC.baseAddress, Int32(columnsC))
          }
        }
      }
    }
    logger.log(message: "BLAS took \(elapsed) seconds")
  }

  private func compareResults(logger: Logger) {
    // Get an UnsafeBufferPointer<Float> to the contents of the MTLBuffer
    // that holds the results of the multiplication.
    let rawPointer = matrixC.data.contents()
    let count = matrixC.rows * matrixC.columns
    let typedPointer = rawPointer.bindMemory(to: Float.self, capacity: count)
    let bufferedPointer = UnsafeBufferPointer(start: typedPointer, count: count)

    // Print the first 10 results, to make sure it's not all 0s or NaNs.
    logger.log(message: "First 10 results:")
    for i in 0..<10 {
      logger.log(message: String(format: "%f %f", arrayC[i], bufferedPointer[i]))
    }

    // Compare the output from MPS with the output from BLAS. Notice that there 
    // will be differences in precision of around 1e-03, probably because Metal
    // uses floats with less precision internally.
    var largestError: Float = 0
    var averageError: Float = 0
    for i in 0..<arrayC.count {
      let error = abs(bufferedPointer[i] - arrayC[i])
      if error > largestError {
        largestError = error
      }
      averageError += error
    }
    averageError /= Float(arrayC.count)

    logger.log(message: "Largest error: \(largestError)")
    logger.log(message: "Average error: \(averageError)")
  }
}
