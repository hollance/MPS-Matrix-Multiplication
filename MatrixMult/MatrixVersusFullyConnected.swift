import Foundation
import MetalPerformanceShaders

/*
  A fully-connected layer performs this calculation: 
  
      fc = W.T * x + b
    
  where W.T is the transposed weight matrix, x is the input to the layer, and 
  b is a vector of bias values.

  We can use MPS Matrix Multiplication to do a very similar calculation:

      C = AxB + C

  Here, A contains the "weight" values, B is the input, and the initial values
  of C serve as the "bias" values.

  Note that B and C are column vectors, so they only have one column.

  Matrix A is the transposed weight matrix, so its number of rows corresponds
  to the number of neurons in the fully-connected layer, and the columns in A
  are equal in number to the number of neurons in the previous layer.
*/
private let rowsA = 8192         // this is the number of neurons in the layer
private let columnsA = 8192      // this is the number of inputs
private let rowsB = columnsA
private let rowsC = rowsA

class MatrixVersusFullyConnected {
  private var device: MTLDevice!
  private var commandQueue: MTLCommandQueue!

  private var matrixMultiplication: MPSMatrixMultiplication!
  private var matrixA: MPSMatrix!
  private var matrixB: MPSMatrix!
  private var matrixC: MPSMatrix!

  private var fc: MPSCNNFullyConnected!
  private var inputImage: MPSImage!
  private var outputImage: MPSImage!

  private var arrayA = [Float](repeating: 0, count: rowsA * columnsA)
  private var arrayB = [Float](repeating: 0, count: rowsB)
  private var arrayC = [Float](repeating: 0, count: rowsC)

  func run(logger: Logger) {
    logger.log(message: "*** Matrix multiplication vs. fully-connected layer ***")
    logger.log(message: "(\(rowsA) x \(columnsA)) * (\(rowsB) x 1) + (\(rowsC) x 1) = (\(rowsC) x 1)")

    randomizeArrays()
    initMatrix()
    initFullyConnected()
    predictMatrix(logger: logger)
    predictFullyConnected(logger: logger)
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

    // Also fill up C with some random numbers, since we use this as the bias
    // vector.
    for i in 0..<arrayC.count {
      arrayC[i] = Float(2*drand48() - 1)
    }
  }

  private func initMatrix() {
    device = MTLCreateSystemDefaultDevice()
    guard device != nil else {
      fatalError("Error: This device does not support Metal")
    }

    guard MPSSupportsMTLDevice(device) else {
      fatalError("Error: This device does not support Metal Performance Shaders")
    }

    commandQueue = device.makeCommandQueue()

    // Note: beta is now 1, so that the initial contents of C are added to AxB.
    matrixMultiplication = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: rowsC, resultColumns: 1, interiorColumns: columnsA, alpha: 1, beta: 1)

    // Note: here we pass arrayC to bufferC instead of using an empty buffer.
    let bufferA = device.makeBuffer(bytes: arrayA, length: rowsA * columnsA * MemoryLayout<Float>.stride, options: [])
    let bufferB = device.makeBuffer(bytes: arrayB, length: rowsB * MemoryLayout<Float>.stride, options: [])
    let bufferC = device.makeBuffer(bytes: arrayC, length: rowsC * MemoryLayout<Float>.stride, options: [])

    let descA = MPSMatrixDescriptor(dimensions: rowsA, columns: columnsA, rowBytes: columnsA * MemoryLayout<Float>.stride, dataType: .float32)
    let descB = MPSMatrixDescriptor(dimensions: rowsB, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)
    let descC = MPSMatrixDescriptor(dimensions: rowsC, columns: 1, rowBytes: MemoryLayout<Float>.stride, dataType: .float32)

    matrixA = MPSMatrix(buffer: bufferA, descriptor: descA)
    matrixB = MPSMatrix(buffer: bufferB, descriptor: descB)
    matrixC = MPSMatrix(buffer: bufferC, descriptor: descC)
  }

  private func initFullyConnected() {
    let fcDesc = MPSCNNConvolutionDescriptor(kernelWidth: 1, kernelHeight: 1, inputFeatureChannels: rowsB, outputFeatureChannels: rowsC, neuronFilter: nil)

    fc = MPSCNNFullyConnected(device: device, convolutionDescriptor: fcDesc, kernelWeights: arrayA, biasTerms: arrayC, flags: .none)

    // The fully-connected layer does not seem to like .float32 as input,
    // so we'll use .float16. This does mean we have to convert our data
    // from 32-bits to 16-bit floats. For output, it seems float32 is OK.
    let inputImgDesc = MPSImageDescriptor(channelFormat: .float16, width: 1, height: 1, featureChannels: rowsB)
    let outputImgDesc = MPSImageDescriptor(channelFormat: .float32, width: 1, height: 1, featureChannels: rowsC)

    inputImage = MPSImage(device: device, imageDescriptor: inputImgDesc)

    // We have to load the contents of arrayB into the input image. This is a
    // 2D texture array of rowsB/4 texture slices, each of size 1x1 pixel (i.e. 
    // 4 float16s per texture). It's a little annoying having to copy our data
    // into MTLTextures but that's just the way it works.
    let input16 = float32to16(&arrayB, count: arrayB.count)
    input16.withUnsafeBufferPointer { ptr in
      for i in 0..<inputImage.texture.arrayLength {
        let region = MTLRegion(origin: MTLOriginMake(0, 0, 0), size: MTLSizeMake(1, 1, 1))
        inputImage.texture.replace(region: region, mipmapLevel: 0, slice: i, withBytes: ptr.baseAddress!.advanced(by: i*4), bytesPerRow: MemoryLayout<Float16>.stride * 4, bytesPerImage: 0)
      }
    }

    outputImage = MPSImage(device: device, imageDescriptor: outputImgDesc)
  }

  private func predictMatrix(logger: Logger) {
    let elapsed = timeIt {
      let commandBuffer = commandQueue.makeCommandBuffer()

      matrixMultiplication.encode(commandBuffer: commandBuffer, leftMatrix: matrixA, rightMatrix: matrixB, resultMatrix: matrixC)

      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    }
    logger.log(message: "Matrix took \(elapsed) seconds")
  }

  private func predictFullyConnected(logger: Logger) {
    let elapsed = timeIt {
      let commandBuffer = commandQueue.makeCommandBuffer()

      fc.encode(commandBuffer: commandBuffer, sourceImage: inputImage, destinationImage: outputImage)

      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    }
    logger.log(message: "FC layer took \(elapsed) seconds")
  }

  private func compareResults(logger: Logger) {
    // Get an UnsafeBufferPointer<Float> to the contents of the MTLBuffer
    // that holds the results of the multiplication.
    let rawPointer = matrixC.data.contents()
    let count = matrixC.rows * matrixC.columns
    let typedPointer = rawPointer.bindMemory(to: Float.self, capacity: count)
    let bufferedPointer = UnsafeBufferPointer(start: typedPointer, count: count)

    // Convert the output of the fully-connected layer from MPSImage to Floats.
    let fcResults = outputImage.toFloatArray()
    assert(fcResults.count == rowsC)

    // Print the first 10 results, to make sure it's not all 0s or NaNs.
    logger.log(message: "First 10 results:")
    for i in 0..<10 {
      logger.log(message: String(format: "%f %f", fcResults[i], bufferedPointer[i]))
    }

    // Make sure the differences in the outputs are not too large.
    var largestError: Float = 0
    var averageError: Float = 0
    for i in 0..<fcResults.count {
      let error = abs(bufferedPointer[i] - fcResults[i])
      if error > largestError {
        largestError = error
      }
      averageError += error
    }
    averageError /= Float(fcResults.count)

    logger.log(message: "Largest error: \(largestError)")
    logger.log(message: "Average error: \(averageError)")
  }
}
