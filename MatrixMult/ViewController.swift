import UIKit
import QuartzCore

func timeIt(_ block: () -> Void) -> CFTimeInterval {
  let startTime = CACurrentMediaTime()
  block()
  return CACurrentMediaTime() - startTime
}

protocol Logger {
  func log(message: String)
}

class ViewController: UIViewController, Logger {

  @IBOutlet weak var textView: UITextView!

  override func viewDidLoad() {
    super.viewDidLoad()
    textView.text = ""
  }

  @IBAction func button1Tapped() {
    MetalVersusBLAS().run(logger: self)
  }

  @IBAction func button2Tapped() {
    MatrixVersusFullyConnected().run(logger: self)
  }

  func log(message: String) {
    let text = textView.text + message + "\n"
    textView.text = text
  }
}
